#include <QApplication>
#include <QThread>
#include <opencv2/core.hpp>
#include <opencv2/highgui.hpp>

#include "object3d.h"
#include "pose_estimator6d.h"

#include "iomanip"
#include <cmath>
#include <filesystem>

using namespace std;
using namespace cv;
namespace fs = std::filesystem;

#define M_PI       3.14159265358979323846

// Global variables for mouse control
bool isDragging = false;
Point2f lastMousePos;
float rotationSpeed = 0.1f;
float translationSpeed = 1.0f;

cv::Mat BlendFrame(const cv::Mat& frame, const cv::Mat& depth, const cv::Mat& rendering, float alpha = 0.2)
{
    // compose the rendering with the current camera image for demo purposes (can be done more efficiently directly in OpenGL)
    Mat result = frame.clone();
    for (int y = 0; y < frame.rows; y++)
    {
        for (int x = 0; x < frame.cols; x++)
        {
            Vec3b color = rendering.at<Vec3b>(y, x);
            if (depth.at<float>(y, x) != 0.0f)
            {
                result.at<Vec3b>(y, x)[0] = result.at<Vec3b>(y, x)[0] * alpha + color[2] * (1.f - alpha);
                result.at<Vec3b>(y, x)[1] = result.at<Vec3b>(y, x)[1] * alpha + color[1] * (1.f - alpha);
                result.at<Vec3b>(y, x)[2] = result.at<Vec3b>(y, x)[2] * alpha + color[0] * (1.f - alpha);
            }
        }
    }

    return result;
}

cv::Mat drawResultOverlay(const vector<Object3D*>& objects, const cv::Mat& frame)
{
    // Render the model with phong shading
    RenderingEngine::Instance()->setLevel(0);
    
    vector<Point3f> colors;
    colors.push_back(Point3f(1.0, 0.5, 0.0));
    RenderingEngine::Instance()->renderShaded(vector<Object3D*>(objects.begin(), objects.end()), GL_FILL, colors, true);
    
    // Download the rendered frames to the CPU
    Mat rendering = RenderingEngine::Instance()->downloadFrame(RenderingEngine::RGB);
    Mat depth = RenderingEngine::Instance()->downloadFrame(RenderingEngine::DEPTH);

    return BlendFrame(frame, depth, rendering);
}

void onMouse(int event, int x, int y, int flags, void* userdata)
{
    Model* model = static_cast<Model*>(userdata);
    
    Matx44f currentPose = model->getPose();
    
    if (event == EVENT_LBUTTONDOWN) {
        isDragging = true;
        lastMousePos = Point2f(x, y);
    }
    else if (event == EVENT_LBUTTONUP) {
        isDragging = false;
    }
    else if (event == EVENT_MOUSEMOVE && isDragging) {
        Point2f currentMousePos(x, y);
        Point2f delta = currentMousePos - lastMousePos;
        
        // Store the current translation
        Vec3f translation(currentPose(0,3), currentPose(1,3), currentPose(2,3));
        
        // Create rotation matrices
        // Horizontal mouse movement (X) rotates around Y axis
        Matx44f rotY = Matx44f::eye();
        float angleY = -delta.x * rotationSpeed * 0.01f;  // Negative for intuitive rotation
        rotY(0,0) = cos(angleY); rotY(0,2) = sin(angleY);
        rotY(2,0) = -sin(angleY); rotY(2,2) = cos(angleY);
        
        // Vertical mouse movement (Y) rotates around X axis
        Matx44f rotX = Matx44f::eye();
        float angleX = delta.y * rotationSpeed * 0.01f;  // Positive for intuitive rotation
        rotX(1,1) = cos(angleX); rotX(1,2) = -sin(angleX);
        rotX(2,1) = sin(angleX); rotX(2,2) = cos(angleX);
        
        // Remove translation from current pose
        currentPose(0,3) = 0;
        currentPose(1,3) = 0;
        currentPose(2,3) = 0;
        
        // Apply rotations around center
        currentPose = rotX * rotY * currentPose;
        
        // Restore translation
        currentPose(0,3) = translation[0];
        currentPose(1,3) = translation[1];
        currentPose(2,3) = translation[2];
        
        model->setPose(currentPose);
        lastMousePos = currentMousePos;
    }
}

void rotateModel(float deltarotX, float deltarotY, float deltarotZ, Model* model)
{
    Matx44f currentPose = model->getPose();

    // Store the current translation
    Vec3f translation(currentPose(0, 3), currentPose(1, 3), currentPose(2, 3));

    // Create rotation matrices
    Matx44f rotY = Matx44f::eye();
    float angleY = -M_PI * deltarotY / 180.f;
    rotY(0, 0) = cos(angleY); rotY(0, 2) = sin(angleY);
    rotY(2, 0) = -sin(angleY); rotY(2, 2) = cos(angleY);

    Matx44f rotX = Matx44f::eye();
    float angleX = M_PI * deltarotX / 180.f;
    rotX(1, 1) = cos(angleX); rotX(1, 2) = -sin(angleX);
    rotX(2, 1) = sin(angleX); rotX(2, 2) = cos(angleX);

    Matx44f rotZ = Matx44f::eye();
    float angleZ = M_PI * deltarotZ / 180.f;
    rotZ(0, 0) = cos(angleZ); rotZ(0, 1) = -sin(angleZ);
    rotZ(1, 0) = sin(angleZ); rotZ(1, 1) = cos(angleZ);

    // Remove translation from current pose
    currentPose(0, 3) = 0;
    currentPose(1, 3) = 0;
    currentPose(2, 3) = 0;

    // Apply rotations around center
    currentPose = rotZ * rotY * rotX * currentPose;

    // Restore translation
    currentPose(0, 3) = translation[0];
    currentPose(1, 3) = translation[1];
    currentPose(2, 3) = translation[2];

    model->setPose(currentPose);
}

void drawAxes(cv::Mat& image, const cv::Matx44f& pose, const cv::Matx33f& K, float scale = 100.0f)
{
    // Define axis endpoints in model space
    std::vector<cv::Point3f> axisPoints = {
        cv::Point3f(0, 0, 0),      // Origin
        cv::Point3f(scale, 0, 0),  // X axis
        cv::Point3f(0, scale, 0),  // Y axis
        cv::Point3f(0, 0, scale)   // Z axis
    };

    // Transform points to camera space
    std::vector<cv::Point3f> transformedPoints;
    for (const auto& point : axisPoints) {
        cv::Vec4f transformed = pose * cv::Vec4f(point.x, point.y, point.z, 1.0f);
        float w = transformed[3];
        if (std::abs(w) > 1e-5f) {
            transformedPoints.push_back(cv::Point3f(transformed[0] / w, transformed[1] / w, transformed[2] / w));
        }
        else {
            transformedPoints.push_back(cv::Point3f(0, 0, -1)); // Mark as invalid
        }
    }

    // Project points to image space
    std::vector<cv::Point2f> projectedPoints;
    for (const auto& point : transformedPoints) {
        float x = point.x / point.z;
        float y = point.y / point.z;
        float u = K(0,0) * x + K(0,2);
        float v = K(1,1) * y + K(1,2);
        projectedPoints.push_back(cv::Point2f(u, v));
    }

    // Draw axes
    cv::line(image, projectedPoints[0], projectedPoints[1], cv::Scalar(0, 0, 255), 2); // X axis - Red
    cv::line(image, projectedPoints[0], projectedPoints[2], cv::Scalar(0, 255, 0), 2); // Y axis - Green
    cv::line(image, projectedPoints[0], projectedPoints[3], cv::Scalar(255, 0, 0), 2); // Z axis - Blue
}

void AlignModelWithFrame(Model* model, const cv::Mat& frame, cv::Matx33f& K, int width, int height, float zNear, float zFar)
{
    RenderingEngine::Instance()->init(K, width, height, zNear, zFar, 4);
    RenderingEngine::Instance()->makeCurrent();

    model->initBuffers();

    cv::namedWindow("3D Model Alignment Window");
    cv::setMouseCallback("3D Model Alignment Window", onMouse, model);

    Mat result = frame.clone();
    int key = waitKey(1);
    
    while (true)
    {
        // Handle keyboard input
        key = waitKey(1);

        // Rotation Controls
        if (key == 'i') rotateModel(2, 0, 0, model);    // Around X Clockwise
        if (key == 'j') rotateModel(-2, 0, 0, model);   // Around X anti-clockwise
        if (key == 'k') rotateModel(0, 2, 0, model);    // Around Y clockwise
        if (key == 'l') rotateModel(0, -2, 0, model);   // Around Y anti-clockwise
        if (key == 'o') rotateModel(0, 0, 2, model);    // Around Z clockwise
        if (key == 'p') rotateModel(0, 0, -2, model);   // Around Z anit-clockwise
        
        Matx44f currentPose = model->getPose();
            
        // Translation controls
        if (key == 'w') currentPose(2,3) += translationSpeed;  // Move forward
        if (key == 's') currentPose(2,3) -= translationSpeed;  // Move backward
        if (key == 'a') currentPose(0,3) -= translationSpeed;  // Move left
        if (key == 'd') currentPose(0,3) += translationSpeed;  // Move right
        if (key == 'q') currentPose(1,3) += translationSpeed;  // Move up
        if (key == 'e') currentPose(1,3) -= translationSpeed;  // Move down
        
        model->setPose(currentPose);
        
        if (key == 32) {  // Space key
            break;
        }

        // Render the model with phong shading
        RenderingEngine::Instance()->setLevel(0);

        vector<Point3f> colors;
        colors.push_back(Point3f(1.0, 0.5, 0.0));
        RenderingEngine::Instance()->renderShaded(model, GL_FILL, colors[0].x, colors[0].y, colors[0].z, true);

        // Download the rendered frame to the CPU
        Mat rendering = RenderingEngine::Instance()->downloadFrame(RenderingEngine::RGB);
        Mat depth = RenderingEngine::Instance()->downloadFrame(RenderingEngine::DEPTH);

        result = BlendFrame(frame, depth, rendering);

        // Draw axes and bounding box for each object
        drawAxes(result, currentPose, K, 20);
        
        // Add control instructions
        putText(result, "Controls:", Point(10, 30), FONT_HERSHEY_SIMPLEX, 0.7, Scalar(255, 255, 255), 2);
        putText(result, "Mouse: Rotate model", Point(10, 60), FONT_HERSHEY_SIMPLEX, 0.7, Scalar(255, 255, 255), 2);
        putText(result, "WASD: Move model", Point(10, 90), FONT_HERSHEY_SIMPLEX, 0.7, Scalar(255, 255, 255), 2);
        putText(result, "QE: Move up/down", Point(10, 120), FONT_HERSHEY_SIMPLEX, 0.7, Scalar(255, 255, 255), 2);
        putText(result, "IJKLOP: Rotate(X/Y/Z)", Point(10, 150), FONT_HERSHEY_SIMPLEX, 0.7, Scalar(255, 255, 255), 2);
        putText(result, "ESC: Exit", Point(10, 180), FONT_HERSHEY_SIMPLEX, 0.7, Scalar(255, 255, 255), 2);
        putText(result, "SPACE: Confirm", Point(10, 210), FONT_HERSHEY_SIMPLEX, 0.7, Scalar(255, 255, 255), 2);

        Vec3f position = model->getPosition();
        Vec3f angles = model->getRotation();
        putText(result, cv::format("Position: %f, %f, %f", position[0], position[1], position[2]), Point(10, 240), FONT_HERSHEY_SIMPLEX, 0.7, Scalar(255, 0, 0), 2);
        putText(result, cv::format("Orientation: %f, %f, %f", angles[0], angles[1], angles[2]), Point(10, 270), FONT_HERSHEY_SIMPLEX, 0.7, Scalar(255, 0, 0), 2);

        cv::imshow("3D Model Alignment Window", result);
    }
    
    model->setInitialPose(model->getPose());

    RenderingEngine::Instance()->doneCurrent();
    cv::destroyWindow("3D Model Alignment Window");
}

void run(int argc, char* argv[])
{
    QApplication a(argc, argv);

    std::string videoPath = argv[1];
    cv::VideoCapture cap(videoPath);
    if (!cap.isOpened()) {
        std::cerr << "Could not open camera\n";
        return;
    }

    int originalWidth = static_cast<int>(cap.get(cv::CAP_PROP_FRAME_WIDTH));
    int originalHeight = static_cast<int>(cap.get(cv::CAP_PROP_FRAME_HEIGHT));

    float scaleX = 1.0f;
    float scaleY = 1.0f;

    // camera image size
    int width = static_cast<int>(originalWidth * scaleX);
    int height = static_cast<int>(originalHeight * scaleY);

    cv::VideoWriter videoWriter;
    if (std::string(argv[5]) == "true") {
        int fourcc = cv::VideoWriter::fourcc('m', 'p', '4', 'v');
        videoWriter.open(argv[3], fourcc, 30, cv::Size(width, height));
        if (!videoWriter.isOpened()) {
            std::cerr << "Error: Could not open video file for writing!" << std::endl;
            return;
        }
    }

    // near and far plane of the OpenGL view frustum
    float zNear = 10;
    float zFar = 10000;

    // camera instrinsics
    Matx33f K_orig = Matx33f(960.44958699785957, 0., 960.42870459140454, 0., 960.1100568684426, 538.04424577726593, 0., 0., 1.);
    Matx33f K = Matx33f(K_orig(0, 0) * scaleX, 0, K_orig(0, 2) * scaleX, 0, K_orig(1, 1) * scaleY, K_orig(1, 2) * scaleY, 0, 0, 1);
    Matx14f distCoeffs = Matx14f(0.0098123116258309918, -0.058968843650705462, -0.0001703912992670547, 5.8586585298937861e-05);

    float ModelScale = 1000.f;                      // Set scale to mm
    Model* model = new Model(argv[2], ModelScale);
    model->InitializeModelPose(K, width, height);

    Mat frame;
    bool success = cap.read(frame);
    AlignModelWithFrame(model, frame, K, width, height, zNear, zFar);

    float modelDistanceFromCamera = cv::norm(model->getPosition());
    vector<float> distances = { modelDistanceFromCamera - 5.f, modelDistanceFromCamera, modelDistanceFromCamera + 5.f };

    vector<Object3D*> objects;
    objects.push_back(new Object3D(model, 0.55f, distances));
    
    PoseEstimator6D* poseEstimator = new PoseEstimator6D(width, height, zNear, zFar, K, distCoeffs, objects);

    // active the OpenGL context for the offscreen rendering engine during pose estimation
    RenderingEngine::Instance()->makeCurrent();

    int timeout = 0;
    bool showHelp = true;
    Matx44f lastCorrectModelPose = model->getPose();

    // Create output directory for captured frames
    std::string outputDir = argv[4];

    int frameNumber = 1;
    while (true)
    {
        bool success = cap.read(frame);
        if (!success) { break; }

        cv::resize(frame, frame, cv::Size(width, height));

        poseEstimator->estimatePoses(frame, false, true);

        if (objects[0]->isTrackingLost()) {
            model->setPose(lastCorrectModelPose);      

            RenderingEngine::Instance()->doneCurrent();
            AlignModelWithFrame(model, frame, K, width, height, zNear, zFar);
            RenderingEngine::Instance()->makeCurrent();

            objects[0]->setTrackingLost(false);
            model->setInitialPose(model->getPose());
            lastCorrectModelPose = model->getPose();
        }
        else {
            lastCorrectModelPose = model->getPose();
        }

        // render the models with the resulting pose estimates ontop of the input image
        Mat result = drawResultOverlay(objects, frame);

        if (showHelp)
        {
            putText(result, "Press '1' to initialize", Point(150, 250), FONT_HERSHEY_DUPLEX, 1.0, Scalar(255, 255, 255), 1);
            putText(result, "or 'c' to quit", Point(205, 285), FONT_HERSHEY_DUPLEX, 1.0, Scalar(255, 255, 255), 1);
        }

        cv::imshow("result", result);
        if (std::string(argv[5]) == "true") { videoWriter.write(result); }

        int key = waitKey(timeout);

        // start/stop tracking the first object
        if (key == (int)'1')
        {
            poseEstimator->toggleTracking(frame, 0, false);
            poseEstimator->estimatePoses(frame, false, true);
            timeout = 1;
            showHelp = !showHelp;
        }
        // reset the system to the initial state
        if (key == (int)'r')
        {
            poseEstimator->reset();
            cap.set(cv::CAP_PROP_POS_FRAMES, 0);
            timeout = 0;
            showHelp = true;
        }
        // Capture and save both frame and rendered model
        if (key == (int)'c')
        {            
            // Save original frame
            std::string framePath = outputDir + "/frame_" + std::to_string(frameNumber) + ".png";
            cv::imwrite(framePath, frame);
            
            // Save rendered model
            std::string modelPath = outputDir + "/model_" + std::to_string(frameNumber) + ".png";
            cv::imwrite(modelPath, result);
            
            frameNumber += 1;
        }
    }

    cap.release();
    if (std::string(argv[5]) == "true") { videoWriter.release(); }
    cv::destroyAllWindows();

    RenderingEngine::Instance()->doneCurrent();

    for (int i = 0; i < objects.size(); i++)
    {
        delete objects[i];
    }
    objects.clear();

    delete poseEstimator;
}

int main(int argc, char* argv[])
{
    const char* defaultArgs[] = {
     ".exe",
    "../data/Compressor/Compressor3.mp4",  // Video file path
    "../data/Compressor/Compressor.obj",   // Model path
    "../data/Compressor/CompressorTracking3.mp4",   // Output file
    "../data/Compressor/",                      // Output Image folder
    "true"                                      // Save output
    };
    //const char* defaultArgs[] = {
    //".exe",
    //"../data/Motor/Motor1.mp4",  // Video file path
    //"../data/Motor/MotorWithDecimation.obj",   // Model path
    //"../data/Motor/MotorTracking2.mp4",
    // "../data/Motor/",
    //"true"                                      // Save output
    //};
    //const char* defaultArgs[] = {
    // ".exe",
    //"../data/RBOTSample/Squirrel.mp4",  // Video file path
    //"../data/RBOTSample/Squirrel.obj",   // Model path
    //"../data/RBOTSample/SampleTracking2.mp4",
    // "../data/RBOTSample/",
    //"true"                                      // Save output
    //};

    int testArgc;
    char** testArgv;

    testArgc = argc > 1 ? argc : 6;
    testArgv = argc > 1 ? argv : const_cast<char**>(defaultArgs);

    run(testArgc, (char**)testArgv);
}