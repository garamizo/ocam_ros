#include <ros/ros.h>
#include <sensor_msgs/TimeReference.h>
#include <sensor_msgs/Imu.h>
#include <sensor_msgs/MagneticField.h>
#include <sensor_msgs/Image.h>
#include <sensor_msgs/CameraInfo.h>
#include <sensor_msgs/image_encodings.h>
#include <sensor_msgs/distortion_models.h>
#include <sensor_msgs/PointCloud2.h>
#include <image_transport/image_transport.h>
#include <camera_info_manager/camera_info_manager.h>
#include <tf/transform_broadcaster.h>
#include <tf/transform_datatypes.h>
#include <dynamic_reconfigure/server.h>
#include <ocams_1cgn/camConfig.h>
#include <iostream>
#include <fstream>
#include <time.h>
#include <boost/thread.hpp>
#include <image_geometry/stereo_camera_model.h>
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/calib3d/calib3d.hpp>
#include <opencv2/core/utility.hpp>
#include <opencv2/ximgproc/disparity_filter.hpp>
#include <opencv2/imgproc.hpp>
// #include <opencv2/xfeatures2d.hpp>
#include <opencv2/features2d.hpp>
#include <sys/stat.h>
#include <unistd.h>
#include "withrobot_camera.hpp"
#include "myahrs_plus.hpp"

using namespace cv;
// using namespace cv::xfeatures2d;
using namespace cv::ximgproc;
using namespace std;

class StereoCamera
{
    Withrobot::Camera* camera;
    Withrobot::camera_format camFormat;

private:

    std::string devPath_;

public:
    int width_;
    int height_;

    StereoCamera(int resolution, double frame_rate): camera(NULL)
    {
        enum_dev_list();

        camera = new Withrobot::Camera(devPath_.c_str());

        if (resolution == 0) { width_ = 1280; height_ = 960;}
        if (resolution == 1) { width_ = 1280; height_ = 720;}
        if (resolution == 2) { width_ = 640; height_  = 480;}
        if (resolution == 3) { width_ = 640; height_  = 360;}
        if (resolution == 4) { width_ = 320; height_  = 240;}

        camera->set_format(width_, height_, Withrobot::fourcc_to_pixformat('Y','U','Y','V'), 1, (unsigned int)frame_rate);

        /*
         * get current camera format (image size and frame rate)
         */
        camera->get_current_format(camFormat);

        camFormat.print();

        /* Withrobot camera start */
        camera->start();
    }

    ~StereoCamera()
    {
        camera->stop();
        delete camera;
    }

    void enum_dev_list()
    {
        /* enumerate device(UVC compatible devices) list */
        std::vector<Withrobot::usb_device_info> dev_list;
        int dev_num = Withrobot::get_usb_device_info_list(dev_list);

        if (dev_num < 1)
        {
            dev_list.clear();

            return;
        }

        for (unsigned int i=0; i < dev_list.size(); i++)
        {
            if (dev_list[i].product == "oCamS-1CGN-U")
            {
                devPath_ = dev_list[i].dev_node;
                return;
            }
        }
    }

    void uvc_control(int exposure, int gain, int blue, int red, bool ae)
    {
        /* Exposure Setting */
        camera->set_control("Exposure (Absolute)", exposure);

        /* Gain Setting */
        camera->set_control("Gain", gain);

        /* White Balance Setting */
        camera->set_control("White Balance Blue Component", blue);
        camera->set_control("White Balance Red Component", red);

        /* Auto Exposure Setting */
        if (ae)
            camera->set_control("Exposure, Auto", 0x3);
        else
            camera->set_control("Exposure, Auto", 0x1);
    }

    bool getImages(cv::Mat &left_image, cv::Mat &right_image, uint32_t &time_stamp) {

        cv::Mat srcImg(cv::Size(camFormat.width, camFormat.height), CV_8UC2);
        cv::Mat dstImg[2];

        uint32_t ts;

        if (camera->get_frame(srcImg.data, camFormat.image_size, 1) != -1)
        {
            // time stamp
            memcpy(&ts, srcImg.data, sizeof(ts));

            cv::split(srcImg, dstImg);

            time_stamp = ts;
            right_image = dstImg[0];
            left_image = dstImg[1];

            return true;
        } else {
            return false;
        }
    }
};


using namespace WithrobotIMU;
class MyAhrsDriverForROS : public iMyAhrsPlus 
{
public:
    SensorData sensor_data_;
private:
    ros::NodeHandle nh_;
    ros::NodeHandle nh_priv_;

    ros::Publisher time_stamp_pub_;
    ros::Publisher imu_data_pub_;
    ros::Publisher imu_mag_pub_;

    tf::TransformBroadcaster broadcaster_;

    Platform::Mutex lock_;
    //SensorData sensor_data_;

    std::string parent_frame_id_;
    std::string frame_id_;
    double linear_acceleration_stddev_;
    double angular_velocity_stddev_;
    double magnetic_field_stddev_;
    double orientation_stddev_;

    void OnSensorData(int sensor_id, SensorData data)
    {
        LockGuard _l(lock_);
        sensor_data_ = data;
        publish_topic();
    }

    void OnAttributeChange(int sensor_id, std::string attribute_name, std::string value)
    {
        printf("OnAttributeChange(id %d, %s, %s)\n", sensor_id, attribute_name.c_str(), value.c_str());
    }

public:
    MyAhrsDriverForROS(std::string port="", int baud_rate=115200)
        : iMyAhrsPlus(port, baud_rate),
          nh_priv_("~")
    {
        // dependent on user device
        nh_priv_.setParam("port", port);
        nh_priv_.setParam("baud_rate", baud_rate);

        // default frame id
        nh_priv_.param("frame_id", frame_id_, std::string("imu_link"));

        // for testing the tf
        nh_priv_.param("parent_frame_id_", parent_frame_id_, std::string("base_link"));

        // defaults obtained experimentally from device
        nh_priv_.param("linear_acceleration_stddev", linear_acceleration_stddev_, -1.0);
        nh_priv_.param("angular_velocity_stddev", angular_velocity_stddev_, -1.0);
        nh_priv_.param("magnetic_field_stddev", magnetic_field_stddev_, -1.0);
        nh_priv_.param("orientation_stddev", orientation_stddev_, -1.0);

        // publisher for streaming
        time_stamp_pub_     = nh_.advertise<sensor_msgs::TimeReference>("imu/timestamp",1);
        imu_data_pub_       = nh_.advertise<sensor_msgs::Imu>("imu/data", 1);
        imu_mag_pub_        = nh_.advertise<sensor_msgs::MagneticField>("imu/mag", 1);

    }

    ~MyAhrsDriverForROS()
    {}

    bool initialize(std::string mode="")
    {
        bool ok = false;

        do
        {
            if(start() == false) break;

            /* IMU mode */
            if(cmd_data_format(mode.c_str()) == false) break;
            printf("IMU initialized: %s\r\n", mode.c_str());
            ok = true;
        } while(0);

        return ok;
    }

    inline void get_data(SensorData& data)
    {
        LockGuard _l(lock_);
        data = sensor_data_;
    }

    inline SensorData get_data()
    {
        LockGuard _l(lock_);
        return sensor_data_;
    }

    void publish_topic()
    {
        uint32_t time_stamp, sec, nsec;

        sensor_msgs::TimeReference time_stamp_msg;
        sensor_msgs::Imu imu_data_msg;
        sensor_msgs::MagneticField imu_magnetic_msg;

        double linear_acceleration_cov = 0.05;
        double angular_velocity_cov    = 0.025;
        double magnetic_field_cov      = -1;
        double orientation_cov         = 0.1;

        imu_data_msg.linear_acceleration_covariance[0] =
                imu_data_msg.linear_acceleration_covariance[4] =
                imu_data_msg.linear_acceleration_covariance[8] = linear_acceleration_cov;

        imu_data_msg.angular_velocity_covariance[0] =
                imu_data_msg.angular_velocity_covariance[4] =
                imu_data_msg.angular_velocity_covariance[8] = angular_velocity_cov;

        imu_data_msg.orientation_covariance[0] =
                imu_data_msg.orientation_covariance[4] =
                imu_data_msg.orientation_covariance[8] = orientation_cov;

        imu_magnetic_msg.magnetic_field_covariance[0] =
                imu_magnetic_msg.magnetic_field_covariance[4] =
                imu_magnetic_msg.magnetic_field_covariance[8] = magnetic_field_cov;

        ros::Time now = ros::Time::now();

        /* time stamp publish */
        time_stamp = sensor_data_.time_stamp;
        sec = (uint32_t)time_stamp/1000;
        nsec = (uint32_t)(time_stamp - sec*1000) * 1e6;

        ros::Time measurement_time(sec, nsec);
        ros::Time time_ref(0, 0);
        time_stamp_msg.header.stamp = measurement_time;
        time_stamp_msg.header.frame_id = frame_id_;
        time_stamp_msg.time_ref = time_ref;
        time_stamp_pub_.publish(time_stamp_msg);


        now = measurement_time;

        imu_data_msg.header.stamp     =
                imu_magnetic_msg.header.stamp = now;

        imu_data_msg.header.frame_id = frame_id_;


        // orientation
        imu_data_msg.orientation.x = float(sensor_data_.quaternion.x) / 16384.;
        imu_data_msg.orientation.y = float(sensor_data_.quaternion.y) / 16384.;
        imu_data_msg.orientation.z = float(sensor_data_.quaternion.z) / 16384.;
        imu_data_msg.orientation.w = float(sensor_data_.quaternion.w) / 16384.;

        // original data used the g unit, convert to m/s^2
        imu_data_msg.linear_acceleration.x     = float(sensor_data_.imu.ax) / 100.;
        imu_data_msg.linear_acceleration.y     = float(sensor_data_.imu.ay) / 100.;
        imu_data_msg.linear_acceleration.z     = float(sensor_data_.imu.az) / 100.;

        // original data used the degree/s unit, convert to radian/s
        imu_data_msg.angular_velocity.x     = float(sensor_data_.imu.gx) / 900.;
        imu_data_msg.angular_velocity.y     = float(sensor_data_.imu.gy) / 900.;
        imu_data_msg.angular_velocity.z     = float(sensor_data_.imu.gz) / 900.;

        // original data used the uTesla unit, convert to Tesla
        imu_magnetic_msg.magnetic_field.x = float(sensor_data_.imu.mx) / 16.;
        imu_magnetic_msg.magnetic_field.y = float(sensor_data_.imu.my) / 16.;
        imu_magnetic_msg.magnetic_field.z = float(sensor_data_.imu.mz) / 16.;

        // publish the IMU data
        imu_data_pub_.publish(imu_data_msg);
        imu_mag_pub_.publish(imu_magnetic_msg);

        // publish tf
        broadcaster_.sendTransform(tf::StampedTransform(tf::Transform(tf::Quaternion(imu_data_msg.orientation.x, imu_data_msg.orientation.y, imu_data_msg.orientation.z, imu_data_msg.orientation.w),
                                                                      tf::Vector3(0.0, 0.0, 0.0)), now, parent_frame_id_, frame_id_));
    }
};


class oCamStereoROS {
private:

    Withrobot::Camera* camera_ros;
    int resolution_;
    double frame_rate_;
    int exposure_, gain_, wb_blue_, wb_red_;
    bool autoexposure_;
    bool show_image_;
    bool config_changed_;

    bool status_ent = false;
    bool status_out = false;
    bool check_status_ent = false;
    bool check_status_out = false;
    bool check_self_ent = false;
    bool check_self_out = false;

    bool check_height = false;
    bool pass_status_ent = false;
    bool pass_status_out = false;

    int cnt_ent = false;
    int cnt_out = false;
    int status_cnt = 0;
    int count_people_ent = 0;
    int count_people_out = 0;

	float min_height = 0.0;

    cv::Mat dispbgr = cv::Mat::zeros(320, 240, CV_8UC1);
    ros::NodeHandle nh;
    std::string left_frame_id_, right_frame_id_;
    StereoCamera* ocams;
    MyAhrsDriverForROS* IMU;

    /* \brief Image to ros message conversion
     * \param img : the image to publish
     * \param encodingType : the sensor_msgs::image_encodings encoding type
     * \param frameId : the id of the reference frame of the image
     * \param t : the ros::Time to stamp the image
     */
    sensor_msgs::ImagePtr imageToROSmsg(cv::Mat img, const std::string encodingType, std::string frameId, ros::Time t)
    {
        sensor_msgs::ImagePtr ptr = boost::make_shared<sensor_msgs::Image>();
        sensor_msgs::Image& imgMessage = *ptr;
        imgMessage.header.stamp = t;
        imgMessage.header.frame_id = frameId;
        imgMessage.height = img.rows;
        imgMessage.width = img.cols;
        imgMessage.encoding = encodingType;
        int num = 1; //for endianness detection
        imgMessage.is_bigendian = !(*(char *) &num == 1);
        imgMessage.step = img.cols * img.elemSize();
        size_t size = imgMessage.step * img.rows;
        imgMessage.data.resize(size);

        if (img.isContinuous())
            memcpy((char*) (&imgMessage.data[0]), img.data, size);
        else {
            uchar* opencvData = img.data;
            uchar* rosData = (uchar*) (&imgMessage.data[0]);
            for (unsigned int i = 0; i < img.rows; i++) {
                memcpy(rosData, opencvData, imgMessage.step);
                rosData += imgMessage.step;
                opencvData += img.step;
            }
        }
        return ptr;
    }

    void publishCamInfo(const ros::Publisher& pub_cam_info, sensor_msgs::CameraInfo& cam_info_msg, ros::Time t)
    {
        cam_info_msg.header.stamp = t;
        pub_cam_info.publish(cam_info_msg);
    }


    /* \brief Publish a cv::Mat image with a ros Publisher
     * \param img : the image to publish
     * \param pub_img : the publisher object to use
     * \param img_frame_id : the id of the reference frame of the image
     * \param t : the ros::Time to stamp the image
     */
    void publishImage(cv::Mat img, image_transport::Publisher &pub_img, std::string img_frame_id, ros::Time t, std::string encoding_id)
    {
        pub_img.publish(imageToROSmsg(img, encoding_id, img_frame_id, t));
    }

    void device_poll() {
        //Reconfigure confidence
        dynamic_reconfigure::Server<ocams_1cgn::camConfig> server;
        dynamic_reconfigure::Server<ocams_1cgn::camConfig>::CallbackType f;
        f = boost::bind(&oCamStereoROS::callback, this ,_1, _2);
        server.setCallback(f);

        // setup publisher stuff
        image_transport::ImageTransport it(nh);
        image_transport::Publisher left_image_pub = it.advertise("stereo/left/image_raw", 1);
        image_transport::Publisher right_image_pub = it.advertise("stereo/right/image_raw", 1);

        ros::Publisher cam_time_stamp_pub = nh.advertise<sensor_msgs::TimeReference>("stereo/timestamp",1);
        ros::Publisher left_cam_info_pub = nh.advertise<sensor_msgs::CameraInfo>("stereo/left/camera_info", 1);
        ros::Publisher right_cam_info_pub = nh.advertise<sensor_msgs::CameraInfo>("stereo/right/camera_info", 1);

        sensor_msgs::TimeReference time_stamp_msg;
        sensor_msgs::CameraInfo left_info, right_info;

        ROS_INFO("Loading from ROS calibration files");

        // get config from the left, right.yaml in config
        camera_info_manager::CameraInfoManager info_manager(nh);

        info_manager.setCameraName("left");
        info_manager.loadCameraInfo( "package://ocams_1cgn/config/calib/left.yaml");
        left_info = info_manager.getCameraInfo();

        info_manager.setCameraName("right");
        info_manager.loadCameraInfo( "package://ocams_1cgn/config/calib/right.yaml");
        right_info = info_manager.getCameraInfo();

        left_info.header.frame_id = left_frame_id_;
        right_info.header.frame_id = right_frame_id_;

        ROS_INFO("Got camera calibration files");

        /******************************************************************************************************************/

        // loop to publish images;
        cv::Mat left_raw, right_raw;
        cv::Mat left_rgb, right_rgb;

        uint32_t time_stamp, sec, nsec;

        while (ros::ok())
        {
            ros::Time now = ros::Time::now();

            if (!ocams->getImages(left_raw, right_raw, time_stamp)) {
                usleep(10);
                continue;
            } else {
                ROS_INFO_ONCE("Success, found camera");
            }

            // /****************** Rectification *****************/
            cv::cvtColor(left_raw, left_rgb, CV_BayerGR2RGB);
            cv::cvtColor(right_raw, right_rgb, CV_BayerGR2RGB);

            /* time stamp publish */
            sec = (uint32_t)time_stamp/1000;
            nsec = (uint32_t)(time_stamp - sec*1000) * 1e6;
            ros::Time measurement_time(sec, nsec);
            ros::Time time_ref(0, 0);
            time_stamp_msg.header.stamp = measurement_time;
            time_stamp_msg.header.frame_id = left_frame_id_;
            time_stamp_msg.time_ref = time_ref;
            cam_time_stamp_pub.publish(time_stamp_msg);

            now = measurement_time;

            if (left_image_pub.getNumSubscribers() > 0) {
                publishImage(left_rgb, left_image_pub, "left_frame", now, sensor_msgs::image_encodings::BGR8);
            }
            if (right_image_pub.getNumSubscribers() > 0) {
                publishImage(right_rgb, right_image_pub, "right_frame", now, sensor_msgs::image_encodings::BGR8);
            }
            if (left_cam_info_pub.getNumSubscribers() > 0) {
                publishCamInfo(left_cam_info_pub, left_info, now);
            }
            if (right_cam_info_pub.getNumSubscribers() > 0) {
                publishCamInfo(right_cam_info_pub, right_info, now);
            }

            if (show_image_) {
                cv::imshow("left", left_rgb);
                cv::waitKey(1);
            }

        }
    }

    void callback(ocams_1cgn::camConfig &config, uint32_t level) {
        ocams->uvc_control(config.exposure, config.gain, config.wb_blue, config.wb_red, config.auto_exposure);
    }


public:
    /**
         * @brief      { function_description }
         *
         * @param[in]  resolution  The resolution
         * @param[in]  frame_rate  The frame rate
     */
    oCamStereoROS() {
        ros::NodeHandle priv_nh("~");

        /* default parameters */
        resolution_ = 2;
        frame_rate_ = 30.0;
        exposure_ = 100;
        gain_ = 150;
        wb_blue_ = 200;
        wb_red_ = 160;
        autoexposure_= false;
        left_frame_id_ = "left_camera";
        right_frame_id_ = "right_camera";
        show_image_ = true;

        /* get parameters */
        priv_nh.getParam("resolution", resolution_);
        priv_nh.getParam("frame_rate", frame_rate_);
        priv_nh.getParam("exposure", exposure_);
        priv_nh.getParam("gain", gain_);
        priv_nh.getParam("wb_blue", wb_blue_);
        priv_nh.getParam("wb_red", wb_red_);
        priv_nh.getParam("left_frame_id", left_frame_id_);
        priv_nh.getParam("right_frame_id", right_frame_id_);
        priv_nh.getParam("show_image", show_image_);
        priv_nh.getParam("auto_exposure", autoexposure_);

        /* initialize the camera */
        ocams = new StereoCamera(resolution_, frame_rate_);
        ocams->uvc_control(exposure_, gain_, wb_blue_, wb_red_, autoexposure_);
        ROS_INFO("Initialized the camera");

        // thread
        boost::shared_ptr<boost::thread> device_poll_thread;
        device_poll_thread = boost::shared_ptr<boost::thread>(new boost::thread(&oCamStereoROS::device_poll, this));
    }

    ~oCamStereoROS() {
        delete ocams;
        delete IMU;
    }
};

int main (int argc, char **argv)
{
    ros::init(argc, argv, "ocams_1cgn");

    ros::NodeHandle nh;
    ros::NodeHandle priv_nh("~");

    std::string port = std::string("/dev/ttyACM0");
    int baud_rate    = 115200;
    std::string imu_mode = std::string("AMGQUA");

    ros::param::get("~port", port);
    ros::param::get("~imu_mode", imu_mode);
    ros::param::get("~baud_rate", baud_rate);

    MyAhrsDriverForROS sensor(port, baud_rate);
    if(sensor.initialize(imu_mode) == false)
    {
        ROS_ERROR("%s\n", "IMU initialize false!\r\n oCamS-1CGN-U sends IMU data through Virtual COM port.\r\n \
                  So, user needs to write following rules into udev rule file like below.\r\n \
                  -------------------------------------------------------------------------------\r\n \
                  $ sudo vi /etc/udev/rules.d/99-ttyacms.rules\r\n \
                  ATTRS{idVendor}==\"04b4\" ATTRS{idProduct}==\"00f9\", MODE=\"0666\", ENV{ID_MM_DEVICE_IGNORE}=\"1\"\r\n \
                  ATTRS{idVendor}==\"04b4\" ATTRS{idProduct}==\"00f8\", MODE=\"0666\", ENV{ID_MM_DEVICE_IGNORE}=\"1\"\r\n \
                  $ sudo udevadm control -R\r\n \
                  -------------------------------------------------------------------------------\r\n");
    }
    oCamStereoROS ocams_ros;

    ros::spin();

    return 0;
}

