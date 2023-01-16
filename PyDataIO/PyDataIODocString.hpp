#ifndef PYDATAIODOCSTRING_HPP
#define PYDATAIODOCSTRING_HPP

constexpr auto _moduleDocString = "Module providing I/O functions to manage image-based data.\n";

//------------------------//
//----- CDataImageIO -----//
//------------------------//
constexpr auto _dataImageIODocString =
        "Class to handle image file formats. It is mainly backed by OpenCV.\n\n";

constexpr auto _ctorDataImageIO =
        "Construct object to load/write image in basic formats.\n\n"
        "Args:\n\n"
        "   arg1 : self\n\n"
        "   arg2 : path to image file\n\n";

constexpr auto _readDataImageDocString =
        "Read image from basic file format (jpg, png, tif, bmp, npz, dicom...). You can consult OpenCV documentation to have a more complete list of compatible formats.\n\n"
        "Returns:\n\n"
        "   numpy array: image loaded\n\n";

constexpr auto _writeDataImageDocString =
        "Write image to basic file format (jpg, png, tif, bmp...). You can consult OpenCV documentation to have a more complete list of compatible formats. Note: DICOM and NPZ formats are not available for writing.\n\n"
        "Args:\n\n"
        "   numpy array: image to save\n\n";

constexpr auto _isImageFormatDocString =
        "Static method to check if the given extension is a compatible image format\n\n"
        "Args:\n\n"
        "   str: file extension (including the dot)\n\n";

//------------------------//
//----- CDataVideoIO -----//
//------------------------//
constexpr auto _dataVideoIODocString =
        "Class to handle video file formats. It is mainly backed by OpenCV.\n\n";

constexpr auto _ctorDataVideoIO =
        "Construct object to load/write video in basic formats.\n\n"
        "Args:\n\n"
        "   arg1 : self\n\n"
        "   arg2 : path to video file\n\n";

constexpr auto _readDataVideoDocString =
        "Read next video frame from basic file format (avi, mp4, webm). You can consult OpenCV documentation to have a more complete list of compatible formats.\n\n"
        "Returns:\n\n"
        "   numpy array: frame read\n\n";

constexpr auto _writeDataVideo1DocString =
        "Write video frame to basic file format (avi, mp4). You can consult OpenCV documentation to have a more complete list of compatible formats.\n\n"
        "Args:\n\n"
        "   numpy array: frame to append\n\n";

constexpr auto _writeDataVideo2DocString =
        "Write video frame to basic file format (avi, mp4). You can consult OpenCV documentation to have a more complete list of compatible formats.\n\n"
        "Args:\n\n"
        "   numpy array: frame to append\n\n"
        "   str: path to video file\n\n";

constexpr auto _stopReadDocString =
        "Stop reading process.\n\n";

constexpr auto _stopWriteDocString =
        "Stop writing process.\n\n";

constexpr auto _waitWriteFinishedDocString =
        "Wait for writing process to finish.\n\n";

constexpr auto _isVideoFormatDocString =
        "Static method to check if the given extension is a compatible video format\n\n"
        "Args:\n\n"
        "   str: file extension (including the dot)\n\n"
        "   bool: if true check only video format, otherwise check also image format for time sequence\n\n"
        "Returns:\n\n"
        "   bool: True if compatible, False otherwise\n\n";

#endif // PYDATAIODOCSTRING_HPP
