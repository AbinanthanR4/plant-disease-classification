import 'dart:async'; // Import for Timer and TimeoutException
import 'dart:convert'; // Import for jsonDecode
import 'dart:io';
import 'package:flutter/material.dart';
import 'package:http_parser/http_parser.dart';
import 'package:image_picker/image_picker.dart';
import 'package:http/http.dart' as http;
import 'package:mime/mime.dart';
import 'package:path/path.dart'; // For basename

void main() {
  runApp(
    MaterialApp(
      home: PlantDiseasePredictor(),
      debugShowCheckedModeBanner: false,
    ),
  );
}

class PlantDiseasePredictor extends StatefulWidget {
  @override
  _PlantDiseasePredictorState createState() => _PlantDiseasePredictorState();
}

class _PlantDiseasePredictorState extends State<PlantDiseasePredictor> {
  File? _image;
  String _prediction = "";
  String _remedy = "";
  String _status = "Please select an image.";
  String _errorDetails = ""; // To store more detailed errors
  bool _isLoading = false; // To show a loading indicator

  final picker = ImagePicker();
  final String apiUrl = "http://localhost:3000/predict";
  final String apiFieldName = 'plantImage'; // Match Express multer field name

  Future<void> pickImage() async {
    if (_isLoading) return;

    try {
      final pickedFile = await picker.pickImage(source: ImageSource.gallery);
      if (pickedFile != null) {
        setState(() {
          _image = File(pickedFile.path);
          _prediction = "";
          _remedy = "";
          _status = "Image selected. Ready to predict.";
          _errorDetails = "";
          _isLoading = false;
        });
      } else {
        setState(() {
          _status = "Image selection cancelled.";
        });
      }
    } catch (e) {
      print("Error picking image: $e");
      setState(() {
        _status = "Error selecting image.";
        _errorDetails = e.toString();
        _isLoading = false;
      });
    }
  }

  Future<void> predictDisease() async {
    if (_image == null) {
      setState(() {
        _status = "Please select an image first.";
        _errorDetails = "";
      });
      return;
    }
    if (_isLoading) return;

    setState(() {
      _status = "Preparing image...";
      _isLoading = true;
      _prediction = "";
      _remedy = "";
      _errorDetails = "";
    });

    final mimeTypeData = lookupMimeType(_image!.path)?.split('/');
    MediaType? contentType;
    if (mimeTypeData != null && mimeTypeData.length == 2) {
      contentType = MediaType(mimeTypeData[0], mimeTypeData[1]);
      print("Determined MIME type: $contentType");
    } else {
      setState(() {
        _status = "Error: Could not determine image type.";
        _isLoading = false;
      });
      return;
    }

    final request = http.MultipartRequest('POST', Uri.parse(apiUrl));

    try {
      request.files.add(
        await http.MultipartFile.fromPath(
          apiFieldName,
          _image!.path,
          contentType: contentType,
          filename: basename(_image!.path),
        ),
      );
    } catch (e) {
      print("Error creating MultipartFile: $e");
      setState(() {
        _status = "Error reading image file.";
        _errorDetails = e.toString();
        _isLoading = false;
      });
      return;
    }

    setState(() {
      _status = "Uploading image to server...";
    });

    try {
      final streamedResponse = await request.send().timeout(
        const Duration(seconds: 90),
      ); // Increased timeout

      setState(() {
        _status = "Processing response from server...";
      });

      final response = await http.Response.fromStream(streamedResponse);
      print("Response Status Code: ${response.statusCode}");
      print("Response Body: ${response.body}");

      if (response.statusCode >= 200 && response.statusCode < 300) {
        try {
          final Map<String, dynamic> data = jsonDecode(response.body);
          if (data.containsKey('error')) {
            setState(() {
              _status = "Server reported an error.";
              _errorDetails =
                  "API Error: ${data['error']}\nDetails: ${data['details'] ?? 'N/A'}";
              _isLoading = false;
            });
          } else {
            setState(() {
              _status = "Prediction complete!";
              _prediction = data['predicted_disease'] ?? "N/A";
              _remedy = data['remedy_suggestion'] ?? "N/A";
              _errorDetails = "";
              _isLoading = false;
            });
          }
        } catch (e) {
          print("JSON Decode Error: $e");
          setState(() {
            _status = "Error: Invalid response format from server.";
            _errorDetails =
                "Could not parse JSON.\nResponse body:\n${response.body}";
            _isLoading = false;
          });
        }
      } else {
        setState(() {
          _status = "Server error: ${response.statusCode}";
          _errorDetails = "Response from server:\n${response.body}";
          _isLoading = false;
        });
      }
    } on TimeoutException catch (_) {
      print("Request timed out.");
      setState(() {
        _status = "Error: Request timed out.";
        _errorDetails = "The server at $apiUrl did not respond in time.";
        _isLoading = false;
      });
    } on SocketException catch (e) {
      print("Network Error: $e");
      setState(() {
        _status = "Error: Could not connect to server.";
        _errorDetails =
            "Check network connection and server address ($apiUrl).\nDetails: $e";
        _isLoading = false;
      });
    } catch (e) {
      print("An unexpected error occurred: $e");
      setState(() {
        _status = "Error: An unexpected error occurred.";
        _errorDetails = e.toString();
        _isLoading = false;
      });
    }
  }

  @override
  Widget build(BuildContext context) {
    return Scaffold(
      appBar: AppBar(title: Text("Plant Disease Predictor")),
      body: Padding(
        padding: const EdgeInsets.all(16.0),
        child: SingleChildScrollView(
          child: Column(
            crossAxisAlignment: CrossAxisAlignment.stretch,
            children: [
              // Image Preview
              Container(
                height: 250,
                decoration: BoxDecoration(
                  color: Colors.grey[200],
                  border: Border.all(color: Colors.grey),
                  borderRadius: BorderRadius.circular(8),
                ),
                child: Center(
                  child:
                      _image != null
                          ? ClipRRect(
                            borderRadius: BorderRadius.circular(7.0),
                            child: Image.file(
                              _image!,
                              fit: BoxFit.contain,
                              errorBuilder:
                                  (context, error, stackTrace) =>
                                      Text('Error loading image preview'),
                            ),
                          )
                          : Text(
                            "No image selected",
                            style: TextStyle(color: Colors.grey[600]),
                          ),
                ),
              ),
              SizedBox(height: 20),

              // Buttons
              ElevatedButton.icon(
                onPressed: _isLoading ? null : pickImage,
                icon: Icon(Icons.image_search),
                label: Text("1. Select Image"),
                style: ElevatedButton.styleFrom(
                  padding: EdgeInsets.symmetric(vertical: 12),
                ),
              ),
              SizedBox(height: 10),
              ElevatedButton.icon(
                onPressed:
                    (_image == null || _isLoading) ? null : predictDisease,
                icon: Icon(Icons.cloud_upload),
                label: Text("2. Predict Disease"),
                style: ElevatedButton.styleFrom(
                  padding: EdgeInsets.symmetric(vertical: 12),
                  backgroundColor: Theme.of(context).primaryColor,
                  foregroundColor: Colors.white,
                ),
              ),
              SizedBox(height: 20),

              // Status and Loading
              if (_isLoading)
                Padding(
                  padding: const EdgeInsets.symmetric(vertical: 10.0),
                  child: Row(
                    mainAxisAlignment: MainAxisAlignment.center,
                    children: [
                      CircularProgressIndicator(),
                      SizedBox(width: 15),
                      Text(_status),
                    ],
                  ),
                )
              else
                Text(
                  "Status: $_status",
                  style: TextStyle(
                    color:
                        _errorDetails.isNotEmpty ? Colors.red : Colors.blueGrey,
                    fontWeight:
                        _errorDetails.isNotEmpty
                            ? FontWeight.bold
                            : FontWeight.normal,
                  ),
                  textAlign: TextAlign.center,
                ),

              // Show detailed error if present
              if (_errorDetails.isNotEmpty && !_isLoading)
                Padding(
                  padding: const EdgeInsets.only(top: 8.0),
                  child: Text(
                    _errorDetails,
                    style: TextStyle(color: Colors.red.shade700, fontSize: 12),
                    textAlign: TextAlign.center,
                  ),
                ),

              Divider(height: 30),

              // Results
              Text(
                "Prediction:",
                style: TextStyle(fontWeight: FontWeight.bold, fontSize: 16),
              ),
              SizedBox(height: 5),
              Text(_prediction.isNotEmpty ? _prediction : "N/A"),
              SizedBox(height: 15),
              Text(
                "Remedy:",
                style: TextStyle(fontWeight: FontWeight.bold, fontSize: 16),
              ),
              SizedBox(height: 5),
              Text(_remedy.isNotEmpty ? _remedy : "N/A"),
              SizedBox(height: 20),
            ],
          ),
        ),
      ),
    );
  }
}