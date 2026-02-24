import 'dart:io';
import 'dart:typed_data';
import 'package:flutter/material.dart';
import 'package:flutter/foundation.dart';
import 'package:camera/camera.dart';
import 'package:image/image.dart' as img;
import 'package:geolocator/geolocator.dart';

void main() async {
  WidgetsFlutterBinding.ensureInitialized();
  runApp(const CyberApp());
}

// --- BACKGROUND WORKER: Rotation, Scaling & Color ---
Future<Uint8List> processFrame(Map<String, dynamic> params) async {
  final CameraImage image = params['image'];
  final int sensorOrientation = params['sensorOrientation'];
  
  try {
    final int srcW = image.width;
    final int srcH = image.height;
    const int targetW = 640;
    const int targetH = 480;

    var rgbImage = img.Image(width: targetW, height: targetH);
    final Uint8List yPlane = image.planes[0].bytes;
    final Uint8List uPlane = image.planes[1].bytes;
    final Uint8List vPlane = image.planes[2].bytes;
    final int uvRowStride = image.planes[1].bytesPerRow;
    final int uvPixelStride = image.planes[1].bytesPerPixel!;

    for (int y = 0; y < targetH; y++) {
      for (int x = 0; x < targetW; x++) {
        int srcX, srcY;
        if (sensorOrientation == 90) {
          srcX = (y * (srcW / targetH)).toInt().clamp(0, srcW - 1);
          srcY = ((targetW - x - 1) * (srcH / targetW)).toInt().clamp(0, srcH - 1);
        } else {
          srcX = (x * (srcW / targetW)).toInt().clamp(0, srcW - 1);
          srcY = (y * (srcH / targetH)).toInt().clamp(0, srcH - 1);
        }

        final int yIndex = srcY * srcW + srcX;
        final int uvIndex = (srcY >> 1) * uvRowStride + (srcX >> 1) * uvPixelStride;

        final int yp = yPlane[yIndex];
        final int up = uPlane[uvIndex.clamp(0, uPlane.length - 1)];
        final int vp = vPlane[uvIndex.clamp(0, vPlane.length - 1)];

        int r = (yp + 1.370705 * (vp - 128)).toInt().clamp(0, 255);
        int g = (yp - 0.337633 * (up - 128) - 0.698001 * (vp - 128)).toInt().clamp(0, 255);
        int b = (yp + 1.732446 * (up - 128)).toInt().clamp(0, 255);

        rgbImage.setPixelRgb(x, y, r, g, b);
      }
    }
    return Uint8List.fromList(img.encodeJpg(rgbImage, quality: 30));
  } catch (e) {
    return Uint8List(0);
  }
}

class CyberApp extends StatelessWidget {
  const CyberApp({super.key});
  @override
  Widget build(BuildContext context) => MaterialApp(
    debugShowCheckedModeBanner: false,
    theme: ThemeData.dark(),
    home: const UplinkScreen(),
  );
}

class UplinkScreen extends StatefulWidget {
  const UplinkScreen({super.key});
  @override
  State<UplinkScreen> createState() => _UplinkScreenState();
}

class _UplinkScreenState extends State<UplinkScreen> {
  CameraController? _controller;
  RawDatagramSocket? _socket;
  bool _isTransmitting = false;
  bool _isProcessing = false;
  int _frameIdCounter = 0;
  int _sensorOrientation = 0;
  
  // Zoom State
  double _currentZoom = 1.0;
  double _minZoom = 1.0;
  double _maxZoom = 1.0;
  double _baseZoom = 1.0;
  
  String _status = "INITIALIZING...";
  String _coords = "AWAITING GPS...";
  double _lastLat = 0.0, _lastLon = 0.0;

  final String _targetIp = "192.168.1.151"; 
  final int _targetPort = 5000;

  @override
  void initState() {
    super.initState();
    _initSystems();
  }

  Future<void> _initSystems() async {
    _socket = await RawDatagramSocket.bind(InternetAddress.anyIPv4, 0);
    final cameras = await availableCameras();
    if (cameras.isEmpty) return;

    final camera = cameras.first;
    _sensorOrientation = camera.sensorOrientation;

    _controller = CameraController(
      camera, 
      ResolutionPreset.low, 
      enableAudio: false,
      imageFormatGroup: ImageFormatGroup.yuv420,
    );
    
    await _controller!.initialize();

    // Setup Zoom Limits
    _minZoom = await _controller!.getMinZoomLevel();
    _maxZoom = await _controller!.getMaxZoomLevel();

    LocationPermission permission = await Geolocator.checkPermission();
    if (permission == LocationPermission.denied) {
      permission = await Geolocator.requestPermission();
    }

    Geolocator.getPositionStream(
      locationSettings: const LocationSettings(accuracy: LocationAccuracy.high, distanceFilter: 1)
    ).listen((pos) {
      _lastLat = pos.latitude; 
      _lastLon = pos.longitude;
      if (mounted) setState(() => _coords = "LAT: ${_lastLat.toStringAsFixed(6)} | LON: ${_lastLon.toStringAsFixed(6)}");
    });

    if (mounted) setState(() => _status = "LINK_READY");
  }

  void _handleZoom(double zoom) {
    double clampedZoom = zoom.clamp(_minZoom, _maxZoom);
    setState(() => _currentZoom = clampedZoom);
    _controller?.setZoomLevel(clampedZoom);
  }

  void _toggleUplink() async {
    if (_isTransmitting) {
      await _controller!.stopImageStream();
      setState(() { _isTransmitting = false; _status = "OFFLINE"; });
      return;
    }

    if (_lastLat == 0.0) {
      setState(() => _status = "WAITING FOR GPS...");
      Position p = await Geolocator.getCurrentPosition();
      _lastLat = p.latitude;
      _lastLon = p.longitude;
    }

    setState(() { _isTransmitting = true; _status = "UPLINK_ACTIVE"; });

    _controller!.startImageStream((CameraImage image) async {
      if (!_isTransmitting || _isProcessing) return;
      _isProcessing = true;

      final double snapLat = _lastLat;
      final double snapLon = _lastLon;

      try {
        final Uint8List jpegBytes = await compute(processFrame, {
          'image': image,
          'sensorOrientation': _sensorOrientation,
        });

        if (jpegBytes.isNotEmpty) {
          final int totalSize = jpegBytes.length;
          final int frameId = _frameIdCounter++;
          final InternetAddress dest = InternetAddress(_targetIp);

          for (int offset = 0; offset < totalSize; offset += 1200) {
            int end = (offset + 1200 < totalSize) ? offset + 1200 : totalSize;
            
            final header = ByteData(28);
            header.setUint32(0, frameId, Endian.big);
            header.setUint32(4, offset, Endian.big);
            header.setUint32(8, totalSize, Endian.big);
            header.setFloat64(12, snapLat, Endian.big);
            header.setFloat64(20, snapLon, Endian.big);

            final builder = BytesBuilder(copy: false);
            builder.add(header.buffer.asUint8List());
            builder.add(jpegBytes.sublist(offset, end));
            _socket?.send(builder.toBytes(), dest, _targetPort);
          }
        }
      } catch (e) {
        debugPrint("TX_ERR: $e");
      } finally {
        _isProcessing = false;
      }
    });
  }

  @override
  Widget build(BuildContext context) {
    if (_controller == null || !_controller!.value.isInitialized) {
      return const Scaffold(backgroundColor: Colors.black, body: Center(child: CircularProgressIndicator()));
    }
    return Scaffold(
      backgroundColor: Colors.black,
      body: Stack(children: [
        // PINCH GESTURE LAYER
        Positioned.fill(
          child: GestureDetector(
            onScaleStart: (details) => _baseZoom = _currentZoom,
            onScaleUpdate: (details) => _handleZoom(_baseZoom * details.scale),
            child: CameraPreview(_controller!),
          ),
        ),
        _buildOverlay(),
        _buildZoomSlider(),
        Align(alignment: const Alignment(0, 0.9), child: _buildButton()),
      ]),
    );
  }

  Widget _buildOverlay() {
    return SafeArea(child: Padding(padding: const EdgeInsets.all(20), child: Column(crossAxisAlignment: CrossAxisAlignment.start, children: [
      Text("SYSTEM: $_status", style: const TextStyle(color: Colors.cyan, fontFamily: 'monospace', fontWeight: FontWeight.bold)),
      Text(_coords, style: const TextStyle(color: Colors.pinkAccent, fontFamily: 'monospace', fontSize: 12)),
      const Spacer(),
      Text("ZOOM: ${_currentZoom.toStringAsFixed(1)}x", style: const TextStyle(color: Colors.yellow, fontFamily: 'monospace')),
    ])));
  }

  Widget _buildZoomSlider() {
    return Positioned(
      right: 10, top: 100, bottom: 200,
      child: RotatedBox(
        quarterTurns: 3,
        child: Slider(
          value: _currentZoom,
          min: _minZoom,
          max: _maxZoom,
          activeColor: Colors.cyan,
          onChanged: _handleZoom,
        ),
      ),
    );
  }

  Widget _buildButton() {
    return InkResponse(
      onTap: _toggleUplink,
      child: Container(
        padding: const EdgeInsets.all(16),
        decoration: BoxDecoration(
          shape: BoxShape.circle,
          border: Border.all(color: _isTransmitting ? Colors.red : Colors.cyan, width: 3),
        ),
        child: Icon(_isTransmitting ? Icons.stop : Icons.sensors, color: Colors.white, size: 40),
      ),
    );
  }

  @override
  void dispose() {
    _controller?.dispose();
    _socket?.close();
    super.dispose();
  }
}