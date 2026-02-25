import 'dart:io';
import 'package:flutter/material.dart';
import 'package:flutter/foundation.dart';
import 'package:camera/camera.dart';
import 'package:image/image.dart' as img;
import 'package:geolocator/geolocator.dart';
import 'package:tflite_flutter/tflite_flutter.dart';

void main() async {
  WidgetsFlutterBinding.ensureInitialized();
  runApp(const CyberApp());
}

class InferenceResult {
  final Uint8List mosaicJpeg;
  final List<Rect> uiBoxes;
  InferenceResult(this.mosaicJpeg, this.uiBoxes);
}

// --- BACKGROUND WORKER (THREAD) ---
Future<InferenceResult> processInference(Map<String, dynamic> params) async {
  final CameraImage image = params['image'];
  final int sensorOrientation = params['sensorOrientation'];
  final Interpreter interpreter = params['interpreter'];
  final double screenW = params['screenWidth'];
  final double screenH = params['screenHeight'];

  try {
    final int width = image.width;
    final int height = image.height;
    var fullRgb = img.Image(width: width, height: height);
    
    // Optimized YUV -> RGB
    final planes = image.planes;
    final yBytes = planes[0].bytes;
    final uBytes = planes[1].bytes;
    final vBytes = planes[2].bytes;
    final int yRowStride = planes[0].bytesPerRow;
    final int uvRowStride = planes[1].bytesPerRow;
    final int uvPixelStride = planes[1].bytesPerPixel!;

    for (int y = 0; y < height; y++) {
      for (int x = 0; x < width; x++) {
        final int uvIndex = (y >> 1) * uvRowStride + (x >> 1) * uvPixelStride;
        final int yIndex = y * yRowStride + x;
        final int yp = yBytes[yIndex];
        final int up = uBytes[uvIndex];
        final int vp = vBytes[uvIndex];

        int r = (yp + ((vp - 128) * 1436 >> 10)).clamp(0, 255);
        int g = (yp - ((up - 128) * 352 >> 10) - ((vp - 128) * 731 >> 10)).clamp(0, 255);
        int b = (yp + ((up - 128) * 1814 >> 10)).clamp(0, 255);
        fullRgb.setPixelRgb(x, y, r, g, b);
      }
    }
    
    if (sensorOrientation == 90) fullRgb = img.copyRotate(fullRgb, angle: 90);

    // YOLO Inference Prep
    img.Image modelInput = img.copyResize(fullRgb, width: 640, height: 640, interpolation: img.Interpolation.nearest);
    final inputBuffer = Float32List(1 * 640 * 640 * 3);
    int pIdx = 0;
    for (var pixel in modelInput) {
      inputBuffer[pIdx++] = pixel.r / 255.0;
      inputBuffer[pIdx++] = pixel.g / 255.0;
      inputBuffer[pIdx++] = pixel.b / 255.0;
    }

    var output = List.filled(1 * 300 * 6, 0.0).reshape([1, 300, 6]);
    interpreter.run(inputBuffer.reshape([1, 640, 640, 3]), output);

    List<img.Image> crops = [];
    List<Rect> uiBoxes = [];
    
    for (var det in output[0]) {
      if (det[4] > 0.45 && det[5] == 0) { // Confidence + Class Face/Person
        // Scaled to UI dimensions immediately in the thread
        uiBoxes.add(Rect.fromLTRB(det[0] * screenW, det[1] * screenH, det[2] * screenW, det[3] * screenH));

        int x1 = (det[0] * fullRgb.width).toInt().clamp(0, fullRgb.width);
        int y1 = (det[1] * fullRgb.height).toInt().clamp(0, fullRgb.height);
        int x2 = (det[2] * fullRgb.width).toInt().clamp(0, fullRgb.width);
        int y2 = (det[3] * fullRgb.height).toInt().clamp(0, fullRgb.height);
        
        int h = (y2 - y1).abs();
        if ((x2 - x1).abs() > 15) {
          // Crop only the top 40% (the head/face)
          crops.add(img.copyCrop(fullRgb, x: x1, y: y1, width: (x2 - x1).abs(), height: (h * 0.4).toInt()));
        }
      }
      if (crops.length >= 16) break; 
    }

    if (crops.isEmpty) return InferenceResult(Uint8List(0), uiBoxes);

    // Mosaic Logic
    int count = crops.length;
    int cols = (count <= 1) ? 1 : (count <= 4) ? 2 : (count <= 9) ? 3 : 4;
    int rows = (count / cols).ceil();
    final int tW = (640 / cols).floor();
    final int tH = (480 / rows).floor();

    img.Image mosaic = img.Image(width: 640, height: 480, backgroundColor: img.ColorRgb8(0, 0, 0));
    for (int i = 0; i < crops.length; i++) {
      img.Image resFace = img.copyResize(crops[i], width: tW, height: tH, interpolation: img.Interpolation.nearest);
      img.compositeImage(mosaic, resFace, dstX: (i % cols) * tW, dstY: (i ~/ cols) * tH);
    }

    return InferenceResult(Uint8List.fromList(img.encodeJpg(mosaic, quality: count > 6 ? 30 : 50)), uiBoxes);
  } catch (e) {
    return InferenceResult(Uint8List(0), []);
  }
}

// --- MAIN UI ---
class CyberApp extends StatelessWidget {
  const CyberApp({super.key});
  @override
  Widget build(BuildContext context) => MaterialApp(debugShowCheckedModeBanner: false, theme: ThemeData.dark(), home: const UplinkScreen());
}

class UplinkScreen extends StatefulWidget {
  const UplinkScreen({super.key});
  @override
  State<UplinkScreen> createState() => _UplinkScreenState();
}

class _UplinkScreenState extends State<UplinkScreen> {
  CameraController? _controller;
  RawDatagramSocket? _socket;
  Interpreter? _interpreter;
  bool _isTransmitting = false;
  bool _isProcessing = false;
  int _frameId = 0;
  List<Rect> _currentBoxes = [];
  double _lat = 0.0, _lon = 0.0;

  @override
  void initState() {
    super.initState();
    _initSystems();
  }

  Future<void> _initSystems() async {
    _interpreter = await Interpreter.fromAsset('assets/models/yolo26n_int8.tflite');
    _socket = await RawDatagramSocket.bind(InternetAddress.anyIPv4, 0);
    final cams = await availableCameras();
    _controller = CameraController(cams.first, ResolutionPreset.low, enableAudio: false);
    await _controller!.initialize();
    Geolocator.getPositionStream().listen((p) => setState(() { _lat = p.latitude; _lon = p.longitude; }));
  }

  void _toggleUplink() {
    setState(() => _isTransmitting = !_isTransmitting);
    if (!_isTransmitting) {
      _controller?.stopImageStream();
      setState(() => _currentBoxes = []);
      return;
    }
    _controller?.startImageStream((img) async {
      if (_isProcessing || !_isTransmitting) return;
      _isProcessing = true;
      
      final Size screen = MediaQuery.of(context).size;
      final res = await compute(processInference, {
        'image': img, 
        'sensorOrientation': _controller!.description.sensorOrientation, 
        'interpreter': _interpreter,
        'screenWidth': screen.width,
        'screenHeight': screen.height,
      });

      if (mounted) {
        setState(() {
          // SIMPLE PREDICTIVE SMOOTHING: 
          // If we had boxes, we LERP them towards the new ones for a smoother adjust
          if (_currentBoxes.length == res.uiBoxes.length) {
            _currentBoxes = List.generate(_currentBoxes.length, (i) {
              return Rect.lerp(_currentBoxes[i], res.uiBoxes[i], 0.6)!; // 60% move to target
            });
          } else {
            _currentBoxes = res.uiBoxes;
          }
        });
        if (res.mosaicJpeg.isNotEmpty) _sendUdp(res.mosaicJpeg);
      }
      _isProcessing = false;
    });
  }

  void _sendUdp(Uint8List data) {
    final dest = InternetAddress("192.168.1.151");
    final int total = data.length;
    final fId = _frameId++;
    for (int i = 0; i < total; i += 1200) {
      int len = (i + 1200 < total) ? 1200 : total - i;
      final packet = Uint8List(28 + len);
      final view = ByteData.view(packet.buffer);
      view.setUint32(0, fId); view.setUint32(4, i); view.setUint32(8, total);
      view.setFloat64(12, _lat); view.setFloat64(20, _lon);
      packet.setRange(28, 28 + len, data, i);
      _socket?.send(packet, dest, 5000);
    }
  }

  @override
  Widget build(BuildContext context) {
    if (_controller == null || !_controller!.value.isInitialized) return const Scaffold(body: Center(child: CircularProgressIndicator()));
    return Scaffold(
      backgroundColor: Colors.black,
      body: Stack(children: [
        CameraPreview(_controller!),
        RepaintBoundary(child: CustomPaint(size: Size.infinite, painter: BoundingBoxPainter(_currentBoxes))),
        _buildHud(),
        Align(alignment: const Alignment(0, 0.9), child: FloatingActionButton(
          backgroundColor: _isTransmitting ? Colors.red : Colors.cyan,
          onPressed: _toggleUplink, child: Icon(_isTransmitting ? Icons.stop : Icons.sensors),
        )),
      ]),
    );
  }

  Widget _buildHud() {
    return SafeArea(child: Padding(padding: const EdgeInsets.all(20), child: Column(crossAxisAlignment: CrossAxisAlignment.start, children: [
      Text("LINK: ${_isTransmitting ? 'ACTIVE' : 'READY'}", style: TextStyle(color: _isTransmitting ? Colors.red : Colors.cyan, fontFamily: 'monospace', fontWeight: FontWeight.bold)),
      Text("GPS: ${_lat.toStringAsFixed(6)}, ${_lon.toStringAsFixed(6)}", style: const TextStyle(fontSize: 10, fontFamily: 'monospace')),
      if (_isTransmitting) Text("TARGETS: ${_currentBoxes.length}", style: const TextStyle(color: Colors.greenAccent, fontWeight: FontWeight.bold)),
    ])));
  }

  @override
  void dispose() { _controller?.dispose(); _socket?.close(); super.dispose(); }
}

class BoundingBoxPainter extends CustomPainter {
  final List<Rect> boxes;
  BoundingBoxPainter(this.boxes);
  @override
  void paint(Canvas canvas, Size size) {
    final paint = Paint()..color = Colors.cyanAccent..style = PaintingStyle.stroke..strokeWidth = 2.0;
    for (var box in boxes) {canvas.drawRect(box, paint);}
  }
  @override
  bool shouldRepaint(covariant BoundingBoxPainter old) => old.boxes != boxes;
}