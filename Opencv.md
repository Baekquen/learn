

# Opencv

## 分类

图像分类   目标检测   图像分割

## 库函数

### Mat    基本图像容器

```cassandra
1 1->Mat() 构造函数:
 2 Mat M(2,2, CV_8UC3, Scalar(0,0,255)); 
 3 int sz[3] = {2,2,2}; 
 4 Mat L(3,sz, CV_8UC(1), Scalar::all(0));
 5 2->Create() function: 函数
 6 M.create(4,4, CV_8UC(2));
 7 3-> 初始化zeros(), ones(), :eyes()矩阵
 8 Mat E = Mat::eye(4, 4, CV_64F);  
 9 Mat O = Mat::ones(2, 2, CV_32F); 
10 Mat Z = Mat::zeros(3,3, CV_8UC1);
11 4->用逗号分隔的初始化函数:
12 Mat C = (Mat_<double>(3,3) << 0, -1, 0, -1, 5, -1, 0, -1, 0);
```

常用操作

```c++
1 Mat A, C;                                 // 只创建信息头部分
2 A = imread(argv[1], CV_LOAD_IMAGE_COLOR); // 这里为矩阵开辟内存
3 Mat B(A);                                 // 使用拷贝构造函数
4 C = A;                                    // 赋值运算符
5 Mat D (A, Rect(10, 10, 100, 100) ); // using a rectangle
6 Mat E = A(Range:all(), Range(1,3)); // using row and column boundaries
7 Mat F = A.clone();
8 Mat G;
9 A.copyTo(G);                       //使用函数 clone() 或者 copyTo() 来拷贝一副图像的矩阵。
```

2、图像基本操作（Mat操作）

2.1 滤波器掩码

滤波器在图像处理中的应用广泛，OpenCV也有个用到了滤波器掩码（也称作核）的函数。使用这个函数，你必须先定义一个表示掩码的 Mat 对象：

```
1 Mat kern = (Mat_<char>(3,3) <<  0, -1,  0,
2                                -1,  5, -1,
3                                 0, -1,  0);
4 filter2D(I, K, I.depth(), kern );
```

2.2 图像混合（addWeighted函数）

 *线性混合操作* 也是一种典型的二元（两个输入）的 *像素操作* ：

这个操可以用来对两幅图像或两段视频产生时间上的 *画面叠化* （cross-dissolve）效果。

```
1 alpha = 0.3；
2 beta = ( 1.0 - alpha );
3 addWeighted( src1, alpha, src2, beta, 0.0, dst);
```

2.3 改变图像的对比度和亮度
两种常用的点过程（即点算子），是用常数对点进行 乘法 和 加法 运算：

```
 1 double alpha;
 2 int beta;
 3 Mat image = imread( argv[1] );
 4 Mat new_image = Mat::zeros( image.size(), image.type() );
 5 for( int y = 0; y < image.rows; y++ )
 6 {
 7     for( int x = 0; x < image.cols; x++ )
 8     {
 9         for( int c = 0; c < 3; c++ )
10         {
11             new_image.at<Vec3b>(y,x)[c] = saturate_cast<uchar>( alpha*( image.at<Vec3b>(y,x)[c] ) + beta );
12         }
13     }
14 }
```

2.4 离散傅立叶变换

  对一张图像使用傅立叶变换就是将它分解成正弦和余弦两部分。也就是将图像从空间域(spatial domain)转换到频域(frequency domain)。 2维图像的傅立叶变换可以用以下数学公式表达:
                         

式中 f 是空间域(spatial domain)值， F 则是频域(frequency domain)值。

```
 1 Mat padded;          //将输入图像延扩到最佳的尺寸
 2 int m = getOptimalDFTSize( I.rows );
 3 int n = getOptimalDFTSize( I.cols ); // 在边缘添加0
 4 copyMakeBorder(I, padded, 0, m - I.rows, 0, n - I.cols, BORDER_CONSTANT, Scalar::all(0));
 5 Mat planes[] = {Mat_<float>(padded), Mat::zeros(padded.size(), CV_32F)};
 6 Mat complexI;
 7 merge(planes, 2, complexI);         // 为延扩后的图像增添一个初始化为0的通道
 8 dft(complexI, complexI);            // 变换结果很好的保存在原始矩阵中
 9 split(complexI, planes);                   // planes[0] = Re(DFT(I), planes[1] = Im(DFT(I))
10 magnitude(planes[0], planes[1], planes[0]);// planes[0] = magnitude
11 Mat magI = planes[0];
```

2.5 基本绘图

```
 1 Point:
 2     Point pt;
 3     pt.x = 10;
 4     pt.y = 8;
 5     或者
 6     Point pt =  Point(10, 8);
 7  
 8 Scalar:
 9     Scalar( B, G, R )   //定义的RGB颜色值为：Blue,Green, Red
10     
11 line 绘直线:
12     line( img,   //输出图像
13         start,   //起始点
14         end,     //结束点
15         Scalar( 0, 0, 0 ),  //颜色
16         thickness=2,    //线条粗细
17         lineType=8 );   //线条类型
18  
19 ellipse 绘椭圆：
20     ellipse( img,   //输出图像
21            Point( w/2.0, w/2.0 ),   //中心为点 (w/2.0, w/2.0) 
22            Size( w/4.0, w/16.0 ),     //大小位于矩形 (w/4.0, w/16.0) 内
23            angle,    //旋转角度为 angle
24            0,
25            360,     //扩展的弧度从 0 度到 360 度
26            Scalar( 255, 0, 0 ),   //颜色
27            thickness,   //线条粗细
28            lineType );    //线条类型
29  
30 circle 绘圆：
31     circle( img,   //输出图像
32          center,    //圆心由点 center 定义
33          w/32.0,     /圆的半径为: w/32.0
34          Scalar( 0, 0, 255 ),   //颜色
35          thickness,   //线条粗细
36          lineType );   //线条类型
37          
38 rectangle 绘矩形：
39     rectangle( rook_image,
40            Point( 0, 7*w/8.0 ),
41            Point( w, w),    //矩形两个对角顶点为 Point( 0, 7*w/8.0 ) 和 Point( w, w)
42            Scalar( 0, 255, 255 ),
43            thickness = -1,
44            lineType = 8 );
45  
46 fillPoly 绘填充的多边形：
47     fillPoly( img,
48             ppt,   //多边形的顶点集为 ppt
49             npt,   //要绘制的多边形顶点数目为 npt
50             1,   //要绘制的多边形数量仅为 1
51             Scalar( 255, 255, 255 ),
52             lineType );
```

2.6 随机数发生器

RNG的实现了一个随机数发生器。 在上面的例子中, rng 是用数值 0xFFFFFFFF 来实例化的一个RNG对象。

```
RNG rng( 0xFFFFFFFF );  
```

------

# *二、imgproc* 模块

#### 1、图像平滑处理

不妨把 *滤波器* 想象成一个包含加权系数的窗口，当使用这个滤波器平滑处理图像时，就把这个窗口滑过图像。

##### 1.1 归一化块滤波器 (Normalized Box Filter)

```c
1 blur( src,    //输入图像
2       dst,    //输出图像
3       Size( i, i ),//定义内核大小( w 像素宽度， h 像素高度)
4       Point(-1,-1))//指定锚点位置(被平滑点)， 如果是负值，取核的中心为锚点。
```

***一般默认point（-1，-1）是中心点****

1.2 高斯滤波器 (Gaussian Filter)

```
1 GaussianBlur( src,    //输入图像
2               dst,     //输出图像
3               Size( i, i ),    //定义内核的大小(需要考虑的邻域范围)。  w 和 h 必须是正奇数，否则将使用  和  参数来计算内核大小。
4               0,    //: x 方向标准方差， 如果是 0 则  使用内核大小计算得到。
5               0 )    //: y 方向标准方差， 如果是 0 则  使用内核大小计算得到。
```

1.3 中值滤波器 (Median Filter)

```
1 medianBlur ( src,   //输入图像
2              dst,    //输出图像
3              i );    //内核大小 (只需一个值，因为我们使用正方形窗口)，必须为奇数。
```

1.4 双边滤波 (Bilateral Filter)

```
1 bilateralFilter ( src,  //输入图像
2  　　　　　　　　　　dst,    //输出图像
3  　　　　　　　　　　i,     //像素的邻域直径
4  　　　　　　　　　　i*2,    //: 颜色空间的标准方差
5  　　　　　　　　　　i/2 );    //: 坐标空间的标准方差(像素单位)
```

#### 2、形态学变换

   形态学操作就是基于形状的一系列图像处理操作。最基本的形态学操作有二：腐蚀与膨胀(Erosion 与 Dilation)。 他们的运用广泛:

消除噪声
分割(isolate)独立的图像元素，以及连接(join)相邻的元素。
寻找图像中的明显的极大值区域或极小值区域。

##### 2.1 腐蚀（Erosion）

   此操作将图像 A 与任意形状的内核 B（通常为正方形或圆形）进行卷积，将内核 B 覆盖区域的最小相素值提取，并代替锚点位置的相素。这一操作将会导致图像中的亮区开始“收缩”。

```c++
 erode(  src,   //原图像
        erosion_dst,    //输出图像
         element );   //腐蚀操作的内核，默认为一个简单的 3x3 矩阵。也可以使用函数 getStructuringElement。
 Mat element = getStructuringElement( erosion_type,
           Size( 2*erosion_size + 1, *erosion_size+1 ),
           Point( erosion_size, erosion_size ) );
```

```c++
  Mat out2;
	Mat structElement2 = getStructuringElement(MORPH_RECT, Size(4,4), Point(-1,-1));
	erode(img,out2,structElement2);
	imshow("腐蚀", out2);imwrite("腐蚀.jpg", out2);

```



##### 2.2 膨胀 (Dilation)

  此操作将图像 A 与任意形状的内核 B（通常为正方形或圆形）进行卷积，将内核 B 覆盖区域的最大相素值提取，并代替锚点位置的相素。这一操作将会导致图像中的亮区开始“扩展”。

```c#
1 ditale( src,   //原图像
2         dilate_dst,    //输出图像
3         element );   //腐蚀操作的内核，默认为一个简单的 3x3 矩阵。也可以使用函数 getStructuringElement。
4 Mat element = getStructuringElement( dilation_type,
5                                      Size( 2*dilation_size + 1, 2*dilation_size+1 ),
6                                      Point( dilation_size, dilation_size ) );
```

```c++
Mat out1;
	Mat structElement1 = getStructuringElement(MORPH_RECT, Size(4,4), Point(-1,-1));
	dilate(img,out1,structElement1);
	imshow("膨胀", out1);imwrite("膨胀.jpg", out1);
```



2.3 开运算 (Opening)

  开运算是通过先对图像腐蚀再膨胀实现的。能够排除小团块物体(假设物体较背景明亮)。

2.4 闭运算(Closing)

  闭运算是通过先对图像膨胀再腐蚀实现的。能够排除小型黑洞(黑色区域)。

2.5 形态梯度(Morphological Gradient)

  膨胀图与腐蚀图之差。能够保留物体的边缘轮廓。

2.6 顶帽(Top Hat)

  原图像与开运算结果图之差。

2.7 黑帽(Black Hat)

  闭运算结果图与原图像之差

3、图像金字塔

  一个图像金字塔是一系列图像的集合 - 所有图像来源于同一张原始图像 - 通过梯次向下采样获得，直到达到某个终止条件才停止采样。有两种类型的图像金字塔常常出现在文献和应用中:
      高斯金字塔(Gaussian pyramid): 用来向下采样

​      拉普拉斯金字塔(Laplacian pyramid): 用来从金字塔低层图像重建上层未采样图像

　　将所有偶数行和列去除。

```
pyrUp( tmp, dst, Size( tmp.cols*2, tmp.rows*2 ) );
```

对图像的向下采样：

- 将图像在每个方向扩大为原来的两倍，新增的行和列以0填充(![0](http://www.opencv.org.cn/opencvdoc/2.3.2/html/_images/math/bc1f9d9bf8a1b606a4188b5ce9a2af1809e27a89.png))
- 使用先前同样的内核(乘以4)与放大后的图像卷积，获得 “新增像素” 的近似值。

```
pyrDown( tmp, dst, Size( tmp.cols/2, tmp.rows/2 ) );
```

4、阈值操作

   最简单的图像分割的方法。 例如：可以将该物体的像素点的灰度值设定为：‘0’（黑色）,其他的像素点的灰度值为：‘255’（白色）；当然像素点的灰度值可以任意，但最好设定的两种颜色对比度较强，方便观察结果

```
 1 threshold( src_gray,    //输入的灰度图像
 2  　　　　　　dst,   //输出图像
 3  　　　　　　threshold_value,   //进行阈值操作时阈值的大小
 4  　　　　　　max_BINARY_value,   //设定的最大灰度值
 5  　　　　　　threshold_type );  // 阈值的类型。
 6                      　　　　　　0: 二进制阈值
 7                      　　　　　　1: 反二进制阈值
 8                      　　　　　　2: 截断阈值
 9                      　　　　　　3: 0阈值
10                      　　　　　　4: 反0阈值
```

#### 5、给图像添加边界

  图像的卷积操作中，处理卷积边缘时需要考虑边缘填充。

```
1 copyMakeBorder( src, 
2         　　 　　dst, 
3         　　 　　top,bottom, left,right, //各边界的宽度
4         　　 　　borderType,     //边界类型，可选择常数边界BORDER_CONSTANT或者复制边界BORDER_REPLICATE。
5        　　　　  value );    //如果 borderType 类型是 BORDER_CONSTANT, 该值用来填充边界像素。
```

6、图像卷积

6.1 函数 filter2D 就可以生成滤波器

```
1 filter2D(src,
2          dst,
3          ddepth,   //dst 的深度。若为负值（如 -1 ），则表示其深度与源图像相等。
4          kernel,   //用来遍历图像的核
5          anchor,   //核的锚点的相对位置，其中心点默认为 (-1, -1) 。
6          delta,   //在卷积过程中，该值会加到每个像素上。默认情况下，这个值为 0 。
7          BORDER_DEFAULT );   //默认即可
```

6.2 Sobel 导数

  Sobel 算子用来计算图像灰度函数的近似梯度。Sobel 算子结合了高斯平滑和微分求导。

​    计算：

1.在两个方向求导:

```
 1 /// 求 X方向梯度
 2   //Scharr( src_gray, grad_x, ddepth, x_order=1, y_order=0, scale, delta, BORDER_DEFAULT );
 3   Sobel( src_gray, grad_x, ddepth, x_order=1, y_order=0, 3, scale, delta, BORDER_DEFAULT );  
 4   convertScaleAbs( grad_x, abs_grad_x );       //将中间结果转换到 CV_8U
 5  
 6   /// 求Y方向梯度
 7   //Scharr( src_gray, grad_y, ddepth, x_order=0, y_order=1, scale, delta, BORDER_DEFAULT );
 8   Sobel( src_gray, grad_y, ddepth, x_order=0, y_order=1, 3, scale, delta, BORDER_DEFAULT );
 9   convertScaleAbs( grad_y, abs_grad_y );
10  
11   /// 合并梯度(近似)
12   addWeighted( abs_grad_x, 0.5, abs_grad_y, 0.5, 0, grad );
```

6.3 Laplace 算子

1. 二阶导数可以用来 *检测边缘*。 因为图像是 “*2维*”, 我们需要在两个方向求导。使用Laplacian算子将会使求导过程变得简单。
2. *Laplacian 算子* 的定义:、

```
1 Laplacian( src_gray,
2            dst,
3            ddepth,        //输出图像的深度。 因为输入图像的深度是 CV_8U 
4            kernel_size,   //内部调用的 Sobel算子的内核大小
5            scale,
6            delta,
7            BORDER_DEFAULT );
```

##### 6.4 Canny 边缘检测

  \1. Canny 使用了滞后阈值，滞后阈值需要两个阈值(高阈值和低阈值):

  \2. 如果某一像素位置的幅值超过 高 阈值, 该像素被保留为边缘像素。
  3.如果某一像素位置的幅值小于 低 阈值, 该像素被排除。
  4.如果某一像素位置的幅值在两个阈值之间,该像素仅仅在连接到一个高于 高 阈值的像素时被保留。


      Canny 推荐的 高:低 阈值比在 2:1 到3:1之间

```
1 Canny(   detected_edges,   //原灰度图像
2          detected_edges,   //输出图像
3          lowThreshold,   //低阈值
4          lowThreshold*ratio,   //设定为低阈值的3倍 (根据Canny算法的推荐)
5          kernel_size );   //设定为 3 (Sobel内核大小，内部使用)
```

6.5 霍夫线变换

1. 霍夫线变换是一种用来寻找直线的方法.
2. 是用霍夫线变换之前, 首先要对图像进行边缘检测的处理，也即霍夫线变换的直接输入只能是边缘二值图像.
3. vascript:void(0);)

```
 1 Canny(src, dst, 50, 200, 3);  //用Canny算子对图像进行边缘检测
 2 vector<Vec2f> lines;
 3 /*标准霍夫线变换*/
 4 HoughLines(  dst,
 5              lines,   //储存着检测到的直线的参数对 (r,\theta) 的容器
 6              rho=1,     //参数极径 rho 以像素值为单位的分辨率. 
 7              CV_PI/180,   //参数极角 \theta 以弧度为单位的分辨率. 我们使用 1度 (即CV_PI/180)
 8              threshold=100,
 9              0,0 );    //srn and stn: 参数默认为0. 
10 //通过画出检测到的直线来显示结果. 
11 for( size_t i = 0; i < lines.size(); i++ )
12 {
13   float rho = lines[i][0], theta = lines[i][1];
14   Point pt1, pt2;
15   double a = cos(theta), b = sin(theta);
16   double x0 = a*rho, y0 = b*rho;
17   pt1.x = cvRound(x0 + 1000*(-b));
18   pt1.y = cvRound(y0 + 1000*(a));
19   pt2.x = cvRound(x0 - 1000*(-b));
20   pt2.y = cvRound(y0 - 1000*(a));
21   line( cdst, pt1, pt2, Scalar(0,0,255), 3, CV_AA);
22 }
23  
24  /*统计概率霍夫线变换*/
25 vector<Vec4i> lines;
26 HoughLinesP( dst,
27              lines,   //储存着检测到的直线的参数对 (x_{start}, y_{start}, x_{end}, y_{end}) 的容器
28              rho=1,    //参数极径 rho 以像素值为单位的分辨率. 
29              CV_PI/180,
30              threshold=50,   //要”检测” 一条直线所需最少的的曲线交点 
31              minLinLength=50,   //能组成一条直线的最少点的数量. 点数量不足的直线将被抛弃.
32              maxLineGap=10 );   //能被认为在一条直线上的亮点的最大距离.
33 //通过画出检测到的直线来显示结果.
34 for( size_t i = 0; i < lines.size(); i++ )
35 {
36   Vec4i l = lines[i];
37   line( cdst, Point(l[0], l[1]), Point(l[2], l[3]), Scalar(0,0,255), 3, CV_AA);
38 }
```

6.6 霍夫圆变换

​    霍夫圆变换的基本原理和上个教程中提到的霍夫线变换类似, 只是点对应的二维极径极角空间被三维的圆心点x, y还有半径r空间取代

```
 1 GaussianBlur( src_gray, src_gray, Size(9, 9), 2, 2 );   //高斯模糊以降低噪声
 2 vector<Vec3f> circles;
 3 HoughCircles( src_gray,   //输入图像 (灰度图)
 4               circles,   //存储下面三个参数: x_{c}, y_{c}, r 集合的容器来表示每个检测到的圆.
 5               CV_HOUGH_GRADIENT,   //指定检测方法. 现在OpenCV中只有霍夫梯度法
 6               dp = 1,   //累加器图像的反比分辨率
 7               min_dist = src_gray.rows/8,   //检测到圆心之间的最小距离
 8               param_1 = 200,    //   //圆心检测阈值.
 9               param_2 = 100,      //圆心检测阈值.
10               0,    //能检测到的最小圆半径,默认为0. 
11               0 );   //能检测到的最大圆半径, 默认为0
12 //绘出检测到的圆
13 for( size_t i = 0; i < circles.size(); i++ )
14 {
15    Point center(cvRound(circles[i][0]), cvRound(circles[i][1]));
16    int radius = cvRound(circles[i][2]);
17    // circle center
18    circle( src, center, 3, Scalar(0,255,0), -1, 8, 0 );
19    // circle outline
20    circle( src, center, radius, Scalar(0,0,255), 3, 8, 0 );
21 }
```

7、 Remapping 重映射

  把一个图像中一个位置的像素放置到另一个图片指定位置的过程。

```
1 remap(   src,
2          dst,
3          map_x,   // x方向的映射参数. 它相当于方法 h(i,j) 的第一个参数
4          map_y,   //y方向的映射参数. 注意 map_y 和 map_x 与 src 的大小一致。
5          CV_INTER_LINEAR,  //非整数像素坐标插值标志. 这里给出的是默认值(双线性插值).
6          BORDER_CONSTANT,   // 默认
7          Scalar(0,0, 0) );
```

8、 仿射变换

  一个任意的仿射变换都能表示为 乘以一个矩阵 (线性变换) 接着再 加上一个向量 (平移).  

  使用矩阵仿射变换。

```
 1 //映射的三个点来定义仿射变换:
 2 srcTri[0] = Point2f( 0,0 );
 3 srcTri[1] = Point2f( src.cols - 1, 0 );
 4 srcTri[2] = Point2f( 0, src.rows - 1 );
 5  
 6 dstTri[0] = Point2f( src.cols*0.0, src.rows*0.33 );
 7 dstTri[1] = Point2f( src.cols*0.85, src.rows*0.25 );
 8 dstTri[2] = Point2f( src.cols*0.15, src.rows*0.7 );
 9  
10 //getAffineTransform 来求出仿射变换
11 warp_mat = getAffineTransform( srcTri, dstTri );
12 warpAffine(  src, 
13              warp_dst,
14              warp_mat,   //仿射变换矩阵
15              warp_dst.size() );   //输出图像的尺寸
```

9、 直方图均衡化

  直方图均衡化是通过拉伸像素强度分布范围来增强图像对比度的一种方法.

```
equalizeHist( src, dst );
```

10、模板匹配

```
matchTemplate( img, templ, result, match_method );
```

------

# *三、feature2d* 模块. 2D特征框架

图像特征类型:

- ​    边缘
- ​    角点 (感兴趣关键点)
- ​    斑点(Blobs) (感兴趣区域)

1、Harris 角点检测子

javascript:void(0);)

```
 1 cornerHarris( src_gray,
 2              dst,
 3              blockSize=2,
 4              apertureSize=3,
 5              k=0.04,
 6              BORDER_DEFAULT );
 7 /// Normalizing
 8 normalize( dst, dst_norm, 0, 255, NORM_MINMAX, CV_32FC1, Mat() );
 9 convertScaleAbs( dst_norm, dst_norm_scaled );
10  
11 /// Drawing a circle around corners
12 for( int j = 0; j < dst_norm.rows ; j++ )
13      { for( int i = 0; i < dst_norm.cols; i++ )
14           {
15             if( (int) dst_norm.at<float>(j,i) > thresh )
16               {
17                circle( dst_norm_scaled, Point( i, j ), 5,  Scalar(0), 2, 8, 0 );
18               }
19           }
20       }
21     /// Showing the result
22     namedWindow( corners_window, CV_WINDOW_AUTOSIZE );
23     imshow( corners_window, dst_norm_scaled );
24 }
```

2、Shi-Tomasi角点检测子

```
 1 /// Apply corner detection
 2 goodFeaturesToTrack( src_gray,
 3                corners,
 4                maxCorners,
 5                qualityLevel,
 6                minDistance,
 7                Mat(),
 8                blockSize,
 9                useHarrisDetector,
10                k );
11  
12 /// Draw corners detected
13 cout<<"** Number of corners detected: "<<corners.size()<<endl;
14 int r = 4;
15 for( int i = 0; i < corners.size(); i++ )
16 {    circle( copy, 
17      corners[i],
18      r,
19      Scalar(rng.uniform(0,255),
20      rng.uniform(0,255),
21      rng.uniform(0,255)), -1, 8, 0 );
22 }
23 /// Show what you got
24 namedWindow( source_window, CV_WINDOW_AUTOSIZE );
25 imshow( source_window, copy );
```

3、特征点检测

```
 1 //--Detect the keypoints using SURF Detector
 2   int minHessian = 400;
 3  
 4   SurfFeatureDetector detector( minHessian );
 5  
 6   std::vector<KeyPoint> keypoints_1, keypoints_2;
 7  
 8   detector.detect( img_1, keypoints_1 );
 9   detector.detect( img_2, keypoints_2 );
10  
11  //-- Draw keypoints
12   Mat img_keypoints_1; Mat img_keypoints_2;
13  
14   drawKeypoints( img_1, keypoints_1, img_keypoints_1, Scalar::all(-1), DrawMatchesFlags::DEFAULT );
15   drawKeypoints( img_2, keypoints_2, img_keypoints_2, Scalar::all(-1), DrawMatchesFlags::DEFAULT );
16  
17  //-- Show detected (drawn) keypoints
18   imshow("Keypoints 1", img_keypoints_1 );
19   imshow("Keypoints 2", img_keypoints_2 );
```

## 图像噪声

图像噪声是指存在于图像数据中的不必要的或多余的干扰信息。噪声的存在严重影响了遥感图像的质量，因此在图像增强处理和分类处理之前，必须予以纠正。 [1] [图像](https://baike.baidu.com/item/图像/773234)中各种妨碍人们对其信息接受的因素即可称为图像噪声 。噪声在理论上可以定义为“不可预测，只能用[概率统计](https://baike.baidu.com/item/概率统计/1486966)方法来认识的[随机误差](https://baike.baidu.com/item/随机误差/10810869)”。因此将图像噪声看成是多维随机过程是合适的，因而描述噪声的方法完全可以借用[随机过程](https://baike.baidu.com/item/随机过程/368895)的描述，即用其概率分布函数和[概率密度](https://baike.baidu.com/item/概率密度)分布函数。

#### 1.椒盐噪声

 椒盐噪声也称为脉冲噪声,是图像中经常见到的一种噪声,它是一种随机出现的白点或者黑点,可能是亮的区域有黑色像素或是在暗的区域有白色像素(或是两者皆有)。盐和胡椒噪声的成因可能是影像讯号受到突如其来的...