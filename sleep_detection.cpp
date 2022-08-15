#include <dlib/opencv.h>
#include <opencv2/highgui/highgui.hpp>
#include <dlib/image_processing/frontal_face_detector.h>
#include <dlib/image_processing/render_face_detections.h>
#include <dlib/image_processing.h>
#include <dlib/gui_widgets.h>
#include <dlib/image_io.h>
#include <vector>
#include <SFML/Audio.hpp>

using namespace dlib;
using namespace std;
using namespace cv;
using namespace std;

image_window win;
shape_predictor sp;
std::vector<cv::Point> righteye;
std::vector<cv::Point> lefteye;
char c;
cv::Point p;
int flag_sleep = 0;
int time_warning = 0;
int flag_warning;
std::vector<int> a;
std::map<std::string, sf::SoundBuffer> buffers;
std::map<std::string, sf::Sound> sounds;

double compute_EAR(std::vector<cv::Point> vec)
{

    double a = cv::norm(cv::Mat(vec[1]), cv::Mat(vec[5]));
    double b = cv::norm(cv::Mat(vec[2]), cv::Mat(vec[4]));
    double c = cv::norm(cv::Mat(vec[0]), cv::Mat(vec[3]));
    //compute EAR
    double ear = (a + b) / (2.0 * c);
    return ear;
}

void playSound(std::string path) {
  if (auto it = sounds.find(path); it == sounds.end()) {
    bool ok = buffers[path].loadFromFile(path);
    if(!ok)
        std::cout << "fail" << endl;
    sounds[path] = sf::Sound{buffers[path]};
  }
  sounds[path].play();
}

int main()
{
    try {
        cv::VideoCapture cap(0);

        if (!cap.isOpened()) {
            cerr << "Unable to connect to camera" << endl;
            return 1;
        }
        cap.set(CAP_PROP_FRAME_WIDTH, 640); //use small resolution for fast processing
        cap.set(CAP_PROP_FRAME_HEIGHT, 480);

        // Load face detection and deserialize  face landmarks model.
        frontal_face_detector detector = get_frontal_face_detector();

        deserialize("/home/ubuntu/Desktop/Study/Freelancer/Task6/models/shape_predictor_68_face_landmarks.dat") >> sp;
        int warning = 0;
        int time_warning = 0;
        // Grab and process frames until the main window is closed by the user.
        while (!win.is_closed()) {
            // Grab a frame
            cv::Mat temp;
            if (!cap.read(temp)) {
                break;
            }

            cv_image<bgr_pixel> cimg(temp);
            full_object_detection shape;
            // Detect faces
            std::vector<rectangle> faces = detector(cimg);
            //cout << "Number of faces detected: " << faces.size() << endl;

            win.clear_overlay();
            win.set_image(cimg);
            
            // Find the pose of each face.
            if (faces.size() > 0) {

                shape = sp(cimg, faces[0]); //work only with 1 face

                for (int b = 36; b < 42; ++b) {
                    p.x = shape.part(b).x();
                    p.y = shape.part(b).y();
                    lefteye.push_back(p);
                }
                for (int b = 42; b < 48; ++b) {
                    p.x = shape.part(b).x();
                    p.y = shape.part(b).y();
                    righteye.push_back(p);
                }
                //Compute Eye aspect ration for eyes
                double right_ear = compute_EAR(righteye);
                double left_ear = compute_EAR(lefteye);
                
                if ((right_ear + left_ear) / 2 < 0.25)
                {
                    flag_warning = 1;
                    flag_sleep += 1;
                    warning +=1;
                    time_warning += 1;
                    int check;
                    //int a[4];
                    if (warning <= 4)
                    {
                        a.push_back(time_warning);
                    }    

                    else if(warning > 4)
                    {  
                        a.erase(a.begin()); 
                        a.push_back(time_warning);
                        std::cout << a[0] << "," << a[1] << "," << a[2] << "," << a[3] << endl;
                        check = int(a[3] - a[0]);
                        std::cout << check << endl;

                        if(check < 30 && check > 6)
                        {
                            cout << "warning" << endl;
                            playSound("/home/ubuntu/Beta.wav");
                        } 
                    }
                    if (flag_sleep >= 15 && check <= 4)
                    {                
                        win.add_overlay(dlib::image_window::overlay_rect(faces[0], rgb_pixel(255, 255, 255), "Sleeping"));
                        cout << "sleep" << endl;
                    }       
                }

                else
                {
                    win.add_overlay(dlib::image_window::overlay_rect(faces[0], rgb_pixel(255, 255, 255), "Not sleeping"));
                    flag_sleep = 0;
                    if(flag_warning = 1)
                    {
                        time_warning += 1;
                    }
                }

                righteye.clear();
                lefteye.clear();

                win.add_overlay(render_face_detections(shape));

                c = (char)waitKey(30);
                if (c == 27)
                    break;
            }
        }
    }
    catch (serialization_error& e) {
        cout << "Fail" << endl;
        cout << endl
             << e.what() << endl;
    }
    catch (exception& e) {
        cout << e.what() << endl;
    }
}
