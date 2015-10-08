#include "opencv2/imgproc/imgproc.hpp"
#include "opencv2/highgui/highgui.hpp"
#include "opencv2/video/tracking.hpp"
#include <iostream>
#include <string>

using namespace cv;
using namespace std;


main(int argc, char* argv[])
{

  VideoCapture cap("visiontraffic.avi"); //Open the video
  if(!cap.isOpened()) //Check if the video is really open
    return -1;

  //Params for Shi-Tomasi corner detection
  int  maxCorners = 100, minDistance = 7, blockSize = 7 ;
  double  qualityLevel = 0.3;
   
  //Params for lucas kanade optical flow
  Size winSize  = Size(15,15);
  int maxLevel = 2;

  //Take first frame and 
  Mat oldFrame, oldGray,  mask;
  vector<Point2f> p0;
  int nMaxFrame = 5;
  int nFrame = nMaxFrame;
  vector<uchar> status;
  vector<Point2f> p1;
  

  cap >> oldFrame;
  cvtColor(oldFrame, oldGray, CV_BGR2GRAY);

  //Create a window
  namedWindow( "frame", WINDOW_AUTOSIZE );
  
  while(1)
    {
      Mat frame, frame_gray, err;
      
      //Get the current frame
      cap >> frame;
      if(frame.empty())
	break;

      cvtColor(frame, frame_gray, CV_BGR2GRAY);
     
      if(nFrame % 5)
      	{
	  goodFeaturesToTrack(oldGray, p0, maxCorners, qualityLevel, minDistance, noArray(), blockSize);
	  nFrame = 0;
	}

      nFrame++;

      
      //Apply the Lucas Kanade methods
      if(!p0.empty())
	{
	  calcOpticalFlowPyrLK(oldGray, frame_gray, p0, p1, status, err, winSize, maxLevel, TermCriteria(TermCriteria::COUNT+TermCriteria::EPS, 10, 0.03));
          	 
	  //Follow the previous point      
	  int iStatus = 0;
	  for(int i = 0; i < p0.size(); ++i)
	    {
	      if(!status[i])
		continue;
	      
	      circle(frame, p1[i], 5, CV_RGB(150, 255, 3), -1);
	      p1[iStatus++] = p1[i]; //Select the good point
		  
	    }
	  
	  p1.resize(iStatus);
	}
      
      //Display the frame
      imshow("frame", frame); 
      int k =  waitKey(10);
      if(k == 27)
	break;

      //Now update the previous frame and previous point
      std::swap(p1, p0);
      cv::swap(oldGray, frame_gray);
    };
  
  return 0;

}
