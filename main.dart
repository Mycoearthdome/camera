import 'dart:io';
import 'dart:isolate';
import 'dart:math' as math;
import 'dart:typed_data';
import 'package:flutter/material.dart';
import 'package:flutter/services.dart';
import 'package:camera/camera.dart';
import 'package:tflite_flutter/tflite_flutter.dart';
import 'package:image/image.dart' as img;
import 'package:geolocator/geolocator.dart';

const int inputSize = 640;
const int numAnchors = 8400;
const int numChannels = 20; // 0-3: box, 4: conf, 5-19: landmarks

void main() async {
  WidgetsFlutterBinding.ensureInitialized();
  runApp(const CyberApp());
}

class WorkerCommand {
  final CameraImage image;
  final double screenW, screenH;    // Main display size
  final bool isFront;
  final Uint8List? modelBytes;
  final double latitude, longitude;
  final double previewW, previewH;  // Camera preview size
  WorkerCommand(
    this.image,
    this.screenW,
    this.screenH,
    this.isFront,
    this.modelBytes,
    this.latitude,
    this.longitude,
    this.previewW,
    this.previewH,
  );
}

class InferenceResult {
  final List<Rect> boxes;
  final List<double> confidences;
  final Uint8List? jpeg;
  final String? stats;
  InferenceResult(this.boxes, this.confidences, {this.jpeg, this.stats});
}

//-----------------------------------------------------
// Helper: Crop & convert a face tile from CameraImage
//-----------------------------------------------------
img.Image cropCameraImage(CameraImage image, _Det det, bool isFront) {
  // Map detection coordinates (640x640 model space) back to original camera resolution
  final scaleX = image.width / inputSize;
  final scaleY = image.height / inputSize;

  final x = (det.x * scaleX).round().clamp(0, image.width - 1);
  final y = (det.y * scaleY).round().clamp(0, image.height - 1);
  final w = (det.w * scaleX).round().clamp(1, image.width - x);
  final h = (det.h * scaleY).round().clamp(1, image.height - y);

  final yP = image.planes[0];
  final uP = image.planes[1];
  final vP = image.planes[2];

  final face = img.Image(width:w, height:h);

  for (int cy = 0; cy < h; cy++) {
    for (int cx = 0; cx < w; cx++) {
      int srcX = x + cx;
      int srcY = y + cy;

      int yIdx = srcY * yP.bytesPerRow + srcX;
      int uvIdx = (srcY >> 1) * uP.bytesPerRow + (srcX >> 1) * (uP.bytesPerPixel ?? 1);

      int yp = yP.bytes[yIdx];
      int up = uP.bytes[uvIdx] - 128;
      int vp = vP.bytes[uvIdx] - 128;

      int r = (yp + (vp * 1436 >> 10)).clamp(0, 255);
      int g = (yp - (up * 352 >> 10) - (vp * 731 >> 10)).clamp(0, 255);
      int b = (yp + (up * 1814 >> 10)).clamp(0, 255);

      face.setPixelRgb(cx, cy, r, g, b);
    }
  }

  // Resize to 160x160
  img.Image tile = img.copyResize(face, width: 160, height: 160);

  // Orientation fix
  if (isFront) {
    tile = img.flipHorizontal(tile);
    tile = img.copyRotate(tile, angle: 270);
  } else {
    tile = img.copyRotate(tile, angle: 90);
  }

  return tile;
}

Size _getScreenPreviewSize(CameraController controller, BuildContext context) {
  final pv = controller.value.previewSize!;
  final screenSize = MediaQuery.of(context).size;

  // Check if portrait mode
  bool isPortrait = screenSize.height > screenSize.width;

  // Swap preview dimensions if needed (camera preview is rotated)
  double w = isPortrait ? pv.height.toDouble() : pv.width.toDouble();
  double h = isPortrait ? pv.width.toDouble() : pv.height.toDouble();

  return Size(w, h);
}

//-----------------------------------------------------
// WORKER (BACKGROUND ISOLATE)
//-----------------------------------------------------
void backgroundInferenceWorker(SendPort mainSendPort) async {
  final port = ReceivePort();
  mainSendPort.send(port.sendPort);

  Interpreter? interpreter;
  Float32List? inputBuffer;
  List<List<List<double>>>? outputBuffer;

  RawDatagramSocket? udp;
  final dest = InternetAddress("192.168.1.151");
  int frameCounter = 0;

  try {
    udp = await RawDatagramSocket.bind(InternetAddress.anyIPv4, 0);
  } catch (e) {
    mainSendPort.send("DEBUG: UDP Bind Error: $e");
  }

  Uint8List? rgbBytes;

  await for (final msg in port) {
    if (msg is! WorkerCommand) continue;

    try {
      // ------------------------
      // Interpreter init (once)
      // ------------------------
      if (interpreter == null) {
        final options = InterpreterOptions();
        if (Platform.isAndroid) {
          try {
            options.addDelegate(GpuDelegateV2());
          } catch (_) {
            mainSendPort.send("DEBUG: GPU failed, using CPU");
          }
        }

        interpreter = Interpreter.fromBuffer(msg.modelBytes!, options: options);

        inputBuffer = Float32List(inputSize * inputSize * 3);
        rgbBytes = Uint8List(inputSize * inputSize * 3);

        outputBuffer = List.generate(
          1,
          (_) => List.generate(
            numChannels,
            (_) => List<double>.filled(numAnchors, 0.0),
          ),
        );

        mainSendPort.send("DEBUG: Pipeline Ready. Shape: [1, $numChannels, $numAnchors]");
      }

      // ------------------------
      // Preprocess
      // ------------------------
      _yuvStandardized(msg.image, inputBuffer!, rgbBytes!, msg.isFront);

      // ------------------------
      // Inference
      // ------------------------
      interpreter.run(
        inputBuffer.reshape([1, inputSize, inputSize, 3]),
        outputBuffer!,
      );

      // ------------------------
      // Decode
      // ------------------------
      final decoded = _decodeYOLOWithConfFixed(
        mainSendPort,
        outputBuffer,
        msg.screenW,
        msg.screenH,
        inputSize.toDouble(),
        inputSize.toDouble(),
        msg.isFront,
        Size(msg.previewW, msg.previewH), // preview size passed in WorkerCommand
      );

      final nmsDets = decoded.rawDets;
      Uint8List? jpeg;

      // ------------------------
      // Mosaic (only if faces)
      // ------------------------
      if (decoded.boxes.isNotEmpty) {
        final mosaic = img.Image(
          width: 640,
          height: 640,
          backgroundColor: img.ColorRgb8(0, 0, 0),
        );

        // Use the preprocessed RGB frame
        final fullFrame = img.Image.fromBytes(
          width: inputSize,
          height: inputSize,
          bytes: rgbBytes.buffer,
          numChannels: 3,
        );

        for (int idx = 0; idx < decoded.boxes.length && idx < 16; idx++) {
          final box = decoded.boxes[idx];

          // Map UI box (screen coordinates) back to model pixels
          int x = (box.left / msg.previewW * inputSize).round().clamp(0, inputSize - 1);
          int y = (box.top / msg.previewH * inputSize).round().clamp(0, inputSize - 1);
          int w = (box.width / msg.previewW * inputSize).round().clamp(1, inputSize - x);
          int h = (box.height / msg.previewH * inputSize).round().clamp(1, inputSize - y);

          var tile = img.copyCrop(fullFrame, x: x, y: y, width: w, height: h);
          tile = img.copyResize(tile, width: 160, height: 160, maintainAspect: true);

          // Orientation
          tile = img.copyRotate(tile, angle: msg.isFront ? 270 : 90);
          if (msg.isFront) tile = img.flipHorizontal(tile);

          img.compositeImage(
            mosaic,
            tile,
            dstX: (idx % 4) * 160,
            dstY: (idx ~/ 4) * 160,
          );
        }

        jpeg = Uint8List.fromList(img.encodeJpg(mosaic, quality: 45));
      }
      // ------------------------
      // Send result
      // ------------------------
      mainSendPort.send(InferenceResult(
        decoded.boxes,
        decoded.confidences,
        jpeg: jpeg,
        stats: "${nmsDets.length} faces",
      ));

      // ------------------------
      // UDP streaming
      // ------------------------
      if (udp != null && jpeg != null) {
        frameCounter++;
        for (int i = 0; i < jpeg.length; i += 1100) {
          final len = (i + 1100 < jpeg.length) ? 1100 : jpeg.length - i;
          final packet = Uint8List(28 + len);
          final view = ByteData.view(packet.buffer);

          view.setUint32(0, frameCounter, Endian.big);
          view.setUint32(4, i, Endian.big);
          view.setUint32(8, jpeg.length, Endian.big);
          view.setFloat64(12, msg.latitude, Endian.big);
          view.setFloat64(20, msg.longitude, Endian.big);

          packet.setRange(28, 28 + len, jpeg, i);
          udp.send(packet, dest, 5000);
        }
      }
    } catch (e) {
      mainSendPort.send("DEBUG: Error: $e");
    } finally {
      // ------------------------
      // Unlock UI processing
      // ------------------------
      mainSendPort.send("UNLOCK");
    }
  }
}

//-----------------------------------------------------
// CORE MATH HELPERS
//-----------------------------------------------------

void _yuvStandardized(
  CameraImage image,
  Float32List floatOut,
  Uint8List byteOut,
  bool isFront,
) {
  final int srcW = image.width;
  final int srcH = image.height;
  final yP = image.planes[0];
  final uP = image.planes[1];
  final vP = image.planes[2];

  // Calculate center crop area
  final int cropSize = math.min(srcW, srcH);
  final int offsetX = (srcW - cropSize) ~/ 2;
  final int offsetY = (srcH - cropSize) ~/ 2;

  int fIdx = 0;
  int bIdx = 0;

  for (int y = 0; y < inputSize; y++) {
    for (int x = 0; x < inputSize; x++) {

      final int srcX = offsetX + (x * cropSize ~/ inputSize);
      final int srcY = offsetY + (y * cropSize ~/ inputSize);

      final int yIdx = srcY * yP.bytesPerRow + srcX;
      final int uvIdx = (srcY >> 1) * uP.bytesPerRow +
          (srcX >> 1) * (uP.bytesPerPixel ?? 1);

      final int yp = yP.bytes[yIdx];
      final int up = uP.bytes[uvIdx] - 128;
      final int vp = vP.bytes[uvIdx] - 128;

      int r = (yp + (vp * 1436 >> 10)).clamp(0, 255);
      int g = (yp - (up * 352 >> 10) - (vp * 731 >> 10)).clamp(0, 255);
      int b = (yp + (up * 1814 >> 10)).clamp(0, 255);

      floatOut[fIdx++] = (r - 127.5) / 127.5;
      floatOut[fIdx++] = (g - 127.5) / 127.5;
      floatOut[fIdx++] = (b - 127.5) / 127.5;

      byteOut[bIdx++] = r;
      byteOut[bIdx++] = g;
      byteOut[bIdx++] = b;
    }
  }
}

// Update the return type to include List<_Det>
class DecodeResult {
  final List<Rect> boxes;
  final List<double> confidences;
  final List<_Det> rawDets; // This is the 640x640 raw data for cropping
  DecodeResult(this.boxes, this.confidences, this.rawDets);
}

// Update the function signature
DecodeResult _decodeYOLOWithConfFixed(
  SendPort mainSendPort,
  List output,
  double screenW,
  double screenH,
  double modelW,
  double modelH,
  bool isFront,
  Size previewSize,
) {
  final dets = <_Det>[];
  final matrix = output[0];

  // Parse detections
  for (int i = 0; i < numAnchors; i++) {
    double conf = matrix[4][i].toDouble();
    if (conf < 0.25) continue;

    double cx = matrix[0][i].toDouble(); // right-side reference in pixels
    double cy = matrix[1][i].toDouble(); // fraction
    double w  = matrix[2][i].toDouble(); // fraction
    double h  = matrix[3][i].toDouble(); // fraction

    dets.add(_Det(cx, cy, w, h, conf));
  }

  final nmsDets = _nms(dets, 0.25);

  final previewW = previewSize.width;
  final previewH = previewSize.height;

  final uiBoxes = <Rect>[];
  final uiConfs = <double>[];

  for (var d in nmsDets) {
    double width  = d.w * previewW;
    double height = d.h * previewH;
    double top    = d.y * previewH;

    // Convert right-side reference to left
    double left = previewW - d.x - width;

    // Mirror for front camera
    if (isFront) {
      left = previewW - left;
    }

    uiBoxes.add(Rect.fromLTWH(left, top, width, height));
    uiConfs.add(d.s);
  }

  mainSendPort.send(
    uiBoxes.isNotEmpty
        ? "DEBUG: first box: left=${uiBoxes[0].left.toStringAsFixed(1)}, "
          "top=${uiBoxes[0].top.toStringAsFixed(1)}, "
          "width=${uiBoxes[0].width.toStringAsFixed(1)}, "
          "height=${uiBoxes[0].height.toStringAsFixed(1)}"
        : "DEBUG: first box UI coords: none"
  );

  return DecodeResult(uiBoxes, uiConfs, nmsDets);
}

//-----------------------------------------------------
// UI AND BOILERPLATE
//-----------------------------------------------------

class CyberApp extends StatelessWidget {
  const CyberApp({super.key});
  @override
  Widget build(BuildContext context) => const MaterialApp(debugShowCheckedModeBanner: false, home: UplinkScreen());
}

class UplinkScreen extends StatefulWidget {
  const UplinkScreen({super.key});
  @override
  State<UplinkScreen> createState() => _UplinkScreenState();
}

class _UplinkScreenState extends State<UplinkScreen> {
  CameraController? _controller;
  SendPort? _worker;
  Uint8List? _model;
  bool _processing = false;
  List<Rect> _boxes = [];
  List<double> _confidences = [];
  double _lat = 0, _lon = 0;
  final List<String> _logs = ["SYSTEM BOOT..."];

  @override
  void initState() {
    super.initState();
    _init();
  }

  void _log(String m) {
    setState(() {
      _logs.insert(0, m);
      if (_logs.length > 30) _logs.removeLast();
    });
  }

Future<void> _init() async {
  try {
    final cameras = await availableCameras();
    final front = cameras.firstWhere(
        (c) => c.lensDirection == CameraLensDirection.front);

    _controller =
        CameraController(front, ResolutionPreset.medium, enableAudio: false);
    await _controller!.initialize();

    final data =
        await rootBundle.load('assets/models/yolov8n-face_int8.tflite');
    _model = data.buffer.asUint8List();
    _log("MODEL: Asset loaded");

    final receive = ReceivePort();
    await Isolate.spawn(backgroundInferenceWorker, receive.sendPort);
    receive.listen((msg) {
      if (msg is SendPort) _worker = msg;
      if (msg == "UNLOCK") _processing = false;
      if (msg is String && msg.startsWith("DEBUG:")) _log(msg);
      if (msg is InferenceResult) {
        setState(() {
          _boxes = msg.boxes;
          _confidences = msg.confidences;
          if (msg.stats != null && msg.boxes.isNotEmpty)
            _log("FACE: ${msg.boxes.length} [${msg.stats}]");
        });
      }
    });

    Geolocator.getPositionStream()
        .listen((pos) => {_lat = pos.latitude, _lon = pos.longitude});

    final previewSize = _getScreenPreviewSize(_controller!, context);

    _controller!.startImageStream((img) {
      if (_processing || _worker == null || _model == null) return;
      _processing = true;

      final screenSize = MediaQuery.of(context).size;
      final isFront =
          _controller!.description.lensDirection == CameraLensDirection.front;

      _worker!.send(WorkerCommand(
        img,
        screenSize.width,
        screenSize.height,
        isFront,
        _model,
        _lat,
        _lon,
        previewSize.width,
        previewSize.height,
      ));
    });
  } catch (e) {
    _log("INIT ERROR: $e");
  }
}

  @override
  Widget build(BuildContext context) {
    if (_controller == null || !_controller!.value.isInitialized) {
      return const Scaffold(
        backgroundColor: Colors.black,
        body: Center(
          child: CircularProgressIndicator(color: Colors.lime),
        ),
      );
    }

    final isFront =
        _controller!.description.lensDirection == CameraLensDirection.front;

    return Scaffold(
      backgroundColor: Colors.black,
      body: Stack(
        children: [

          // --------------------------------------------------
          // Camera Layer (DO NOT MODIFY GEOMETRY)
          // --------------------------------------------------
          // Camera + Boxes (LOCKED TO SAME SIZE)
          LayoutBuilder(
            builder: (context, constraints) {
              final previewSize =
                  Size(constraints.maxWidth, constraints.maxHeight);

              return SizedBox(
                width: previewSize.width,
                height: previewSize.height,
                child: Stack(
                  fit: StackFit.expand,
                  children: [
                    FittedBox(
                      fit: BoxFit.cover,
                      child: SizedBox(
                        width: _controller!.value.previewSize!.height,
                        height: _controller!.value.previewSize!.width,
                        child: Transform(
                          alignment: Alignment.center,
                          transform: _controller!.description.lensDirection ==
                                  CameraLensDirection.front
                              ? Matrix4.rotationY(math.pi)
                              : Matrix4.identity(),
                          child: CameraPreview(_controller!),
                        ),
                      ),
                    ),

                    // Boxes now perfectly aligned
                    CustomPaint(
                      size: previewSize,
                      painter: _BoxPainter(_boxes, _confidences),
                    ),
                  ],
                ),
              );
            },
          ),

          // --------------------------------------------------
          // Top gradient for readability
          // --------------------------------------------------
          IgnorePointer(
            child: Container(
              height: 120,
              decoration: const BoxDecoration(
                gradient: LinearGradient(
                  begin: Alignment.topCenter,
                  end: Alignment.bottomCenter,
                  colors: [
                    Colors.black87,
                    Colors.transparent,
                  ],
                ),
              ),
            ),
          ),

          // --------------------------------------------------
          // Controls (Safe area)
          // --------------------------------------------------
          SafeArea(
            child: Padding(
              padding: const EdgeInsets.symmetric(horizontal: 16, vertical: 8),
              child: Row(
                mainAxisAlignment: MainAxisAlignment.spaceBetween,
                children: [

                  // Mode chip
                  Container(
                    padding:
                        const EdgeInsets.symmetric(horizontal: 12, vertical: 6),
                    decoration: BoxDecoration(
                      color: Colors.black54,
                      borderRadius: BorderRadius.circular(20),
                      border: Border.all(color: Colors.cyanAccent, width: 1),
                    ),
                    child: Text(
                      isFront ? "SELFIE MODE" : "REAR MODE",
                      style: const TextStyle(
                        color: Colors.cyanAccent,
                        fontSize: 11,
                        fontWeight: FontWeight.bold,
                        letterSpacing: 1.2,
                      ),
                    ),
                  ),

                  // Toggle button
                  _buildCyberButton(
                    icon: Icons.flip_camera_ios,
                    label: "SWITCH",
                    onTap: _toggleCamera,
                  ),
                ],
              ),
            ),
          ),

          // --------------------------------------------------
          // Bottom Log Console (glass style)
          // --------------------------------------------------
          Positioned(
            left: 0,
            right: 0,
            bottom: 0,
            child: Container(
              height: 120,
              decoration: const BoxDecoration(
                gradient: LinearGradient(
                  begin: Alignment.topCenter,
                  end: Alignment.bottomCenter,
                  colors: [
                    Colors.transparent,
                    Colors.black87,
                  ],
                ),
                border: Border(
                  top: BorderSide(color: Colors.lime, width: 1),
                ),
              ),
              child: ListView.builder(
                reverse: true,
                padding: const EdgeInsets.all(8),
                itemCount: _logs.length,
                itemBuilder: (c, i) => Text(
                  "> ${_logs[i]}",
                  style: const TextStyle(
                    color: Colors.lime,
                    fontSize: 9,
                    fontFamily: 'monospace',
                  ),
                ),
              ),
            ),
          ),
        ],
      ),
    );
  }


Future<void> _toggleCamera() async {
  if (_controller == null) return;

  final cameras = await availableCameras();
  final newDirection = _controller!.description.lensDirection ==
          CameraLensDirection.front
      ? CameraLensDirection.back
      : CameraLensDirection.front;

  final newCamera =
      cameras.firstWhere((c) => c.lensDirection == newDirection);

  await _controller!.stopImageStream();
  await _controller!.dispose();

  setState(() {
    _controller =
        CameraController(newCamera, ResolutionPreset.medium, enableAudio: false);
    _boxes = [];
  });

  await _controller!.initialize();

  final previewSize = _getScreenPreviewSize(_controller!, context);

  _controller!.startImageStream((img) {
    if (_processing || _worker == null || _model == null) return;
    _processing = true;

    final screenSize = MediaQuery.of(context).size;
    final isFront =
        _controller!.description.lensDirection == CameraLensDirection.front;

    _worker!.send(WorkerCommand(
      img,
      screenSize.width,
      screenSize.height,
      isFront,
      _model,
      _lat,
      _lon,
      previewSize.width,
      previewSize.height,
    ));
  });

  _log("SENSOR: Switched to ${newDirection.name.toUpperCase()}");
}

  Widget _buildCyberButton({
    required IconData icon,
    required String label,
    required VoidCallback onTap,
  }) {
    return GestureDetector(
      onTap: onTap,
      child: Container(
        width: 56,
        height: 56,
        decoration: BoxDecoration(
          color: Colors.black54,
          shape: BoxShape.circle,
          border: Border.all(color: Colors.cyanAccent, width: 1),
          boxShadow: const [
            BoxShadow(
              color: Colors.cyanAccent,
              blurRadius: 6,
              spreadRadius: -2,
            ),
          ],
        ),
        child: Icon(icon, color: Colors.cyanAccent),
      ),
    );
  }
}

List<_Det> _nms(List<_Det> dets, double th) {
  dets.sort((a, b) => b.s.compareTo(a.s));
  final keep = <_Det>[];
  for (var d in dets) {
    bool ok = true;
    for (var k in keep) { if (_iou(d, k) > th) ok = false; }
    if (ok) keep.add(d);
  }
  return keep;
}

double _iou(_Det a, _Det b) {
  double x1 = math.max(a.x, b.x);
  double y1 = math.max(a.y, b.y);
  double x2 = math.min(a.x + a.w, b.x + b.w);
  double y2 = math.min(a.y + a.h, b.y + b.h);
  double inter = math.max(0, x2 - x1) * math.max(0, y2 - y1);
  double union = a.w * a.h + b.w * b.h - inter;
  return union == 0 ? 0 : inter / union;
}

class _BoxPainter extends CustomPainter {
  final List<Rect> boxes;
  final List<double> confidences;
  final bool isFront;
  final double screenWidth;

  _BoxPainter(this.boxes, this.confidences, {this.isFront = false, this.screenWidth = 0});

  @override
  void paint(Canvas canvas, Size size) {
    final linePaint = Paint()
      ..color = Colors.cyanAccent
      ..style = PaintingStyle.stroke
      ..strokeWidth = 2.0;

    final fillPaint = Paint()
      ..color = Colors.cyanAccent.withOpacity(0.1)
      ..style = PaintingStyle.fill;

    for (int i = 0; i < boxes.length; i++) {
      Rect rect = boxes[i];

      canvas.drawRect(rect, fillPaint);

      double len = rect.width * 0.2;

      // Top Left
      canvas.drawLine(rect.topLeft, rect.topLeft + Offset(len, 0), linePaint);
      canvas.drawLine(rect.topLeft, rect.topLeft + Offset(0, len), linePaint);

      // Top Right
      canvas.drawLine(rect.topRight, rect.topRight + Offset(-len, 0), linePaint);
      canvas.drawLine(rect.topRight, rect.topRight + Offset(0, len), linePaint);

      // Bottom Left
      canvas.drawLine(rect.bottomLeft, rect.bottomLeft + Offset(len, 0), linePaint);
      canvas.drawLine(rect.bottomLeft, rect.bottomLeft + Offset(0, -len), linePaint);

      // Bottom Right
      canvas.drawLine(rect.bottomRight, rect.bottomRight + Offset(-len, 0), linePaint);
      canvas.drawLine(rect.bottomRight, rect.bottomRight + Offset(0, -len), linePaint);

      // Confidence tag
      final tp = TextPainter(
        text: TextSpan(
          text: "${(confidences[i] * 100).toStringAsFixed(0)}%",
          style: const TextStyle(
            color: Colors.black,
            fontSize: 9,
            fontWeight: FontWeight.bold,
            backgroundColor: Colors.cyanAccent,
            fontFamily: 'monospace',
          ),
        ),
        textDirection: TextDirection.ltr,
      )..layout();
      tp.paint(canvas, Offset(rect.left, rect.top - 12));
    }
  }

  @override
  bool shouldRepaint(covariant _BoxPainter old) => true;
}

class _Det { double x, y, w, h, s; _Det(this.x, this.y, this.w, this.h, this.s); }
class Tuple2<T1, T2> { final T1 item1; final T2 item2; Tuple2(this.item1, this.item2); }