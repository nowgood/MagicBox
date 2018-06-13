#include <opencv2/opencv.hpp>
using namespace std;
using namespace cv;//包含cv命名空间
int main()
{

    Mat img = imread("/home/wangbin/CLionProjects/testopencv/test.jpeg");//imread()函数载入图像
    //从文件中读入图像，注意图像路径最好不要用相对路径，因为CLion生成的exe不在当前目录下。
    if(img.empty())
    {
        fprintf(stderr, "Can not load image\n");//如果读入图像失败，返回错误信息
        return -1;
    }
    //显示图像
    imshow("image", img);//imshow()函数显示图像
    waitKey();//此函数等待按键，按键盘任意键就返回
    return 0;
}
