#include <cv.h>

#include<opencv2/opencv.hpp>

using namespace std;
using namespace cv;
/*使用容器vector和ROI方法完成一个窗口显示多张图片
Images是vector<Mat>,可有push_back方法往里面加若干图片
dst为输出图像，即将所有图片放在一张图片中
imgRows是每行显示几张图片，比如你显示4张图片填2则显示2*2，显示9张填3显示3*3，可以自由选择
示例代码：
Mat img_1 = imread("1.jpg");
Mat img_2 = imread("2.jpg");
Mat img_3 = imread("3.jpg");
Mat img_4 = imread("4.jpg");
Mat dst;
vector<Mat> manyimgV;
manyimgV.push_back(img_1);
manyimgV.push_back(img_2);
manyimgV.push_back(img_3);
manyimgV.push_back(img_4);
ManyImages(manyimgV, dst,2);
imshow("ManyImagesInWindow", dst);
waitKey(0);

备注：运行速度：约960ms，95%时间用在resize上
*/
void ManyImages(vector<Mat> Images, Mat& dst, int imgRows)
{
    int Num = Images.size();//得到Vector容器中图片个数
    //设定包含这些图片的窗口大小，这里都是BGR3通道，如果做灰度单通道，稍微改一下下面这行代码就可以
    Mat Window(300 * ((Num - 1) / imgRows + 1), 300 * imgRows, CV_8UC3, Scalar(0, 0, 0));
    Mat Std_Image;//存放标准大小的图片
    Mat imageROI;//图片放置区域
    Size Std_Size = Size(300, 300);//每个图片显示大小300*300
    int x_Begin = 0;
    int y_Begin = 0;
    for (int i = 0; i < Num; i++)
    {
        x_Begin = (i % imgRows)*Std_Size.width;//每张图片起始坐标
        y_Begin = (i / imgRows)*Std_Size.height;
        resize(Images[i], Std_Image, Std_Size, 0, 0, INTER_LINEAR);//将图像设为标准大小
        //将其贴在Window上
        imageROI = Window(Rect(x_Begin, y_Begin, Std_Size.width, Std_Size.height));
        Std_Image.copyTo(imageROI);
    }
    dst = Window;
}
int main() {
    Mat img_1 = imread("/home/wangbin/CLionProjects/testopencv/test.jpeg");
    Mat img_2 = imread("/home/wangbin/CLionProjects/testopencv/test.jpeg");
    Mat dst;
    vector<Mat> manyimgV;
    manyimgV.push_back(img_1);
    manyimgV.push_back(img_2);
    manyimgV.push_back(img_1);
    manyimgV.push_back(img_2);
    ManyImages(manyimgV, dst, 2);
    imshow("ManyImagesInWindow", dst);
    waitKey(0);
}
