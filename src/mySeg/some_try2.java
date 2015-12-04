package mySeg;

import ij.IJ;
import weka.classifiers.functions.SMO;
import trainableSegmentation.*;
import ij.ImagePlus;
import ij.gui.Roi;
import ij.process.ImageProcessor;
import weka.classifiers.trees.J48;
import hr.irb.fastRandomForest.FastRandomForest;

public class some_try2 {
	
	public static void main(String[] args){
//	new ImageJ();
//    ImagePlus imp = IJ.openImage("C:\\Documents and Settings\\Administrator\\桌面\\1.png");
//    IJ.runPlugIn(image, "fiji.My_Beautiful_Plugin", "parameter=Hello");
//    imp.show();
//    WindowManager.addWindow(imp.getWindow());
// final String classifier_name="C:\\Documents and Settings\\Administrator\\桌面\\1_png_classifier.model";
final Boolean flag = true;
if(flag)
{
    ImagePlus imp = IJ.openImage( "C:\\Documents and Settings\\Administrator\\桌面\\1.png" );
 // create Weka Segmentation object
    WekaSegmentation segmentator = new WekaSegmentation(imp);
    
//    WindowManager.addWindow(imp.getWindow());

 // add pixels to first class (0) from ROI in slice # 1
//    可以选择各种形状的区域，选取像素点，详见Roi
//    用法：public void addExample(int classNum, Roi roi, int n)
//    segmentator.addExample( 0, new Roi( 229, 124, 40, 26 ), 1 );
    // add pixels to second class (1) from ROI in slice # 1
//    segmentator.addExample( 1, new Roi( 70, 200, 500, 50 ), 1 );
    
 // open binary label image
//    ImagePlus labels = IJ.openImage( "C:\\Documents and Settings\\Administrator\\桌面\\1_label.png" );
    // for the first slice, add white pixels as labels for class 2 and 
    // black pixels as labels for class 1
//    segmentator.addBinaryData( labels, 0, "class 2", "class 1");
    
 // create new SMO classifier (default parameters)
 //SMO classifier1 = new SMO();
 J48 classifier2 = new J48();
 FastRandomForest classifier3 = new FastRandomForest();
 
 // assign classifier to segmentator
 segmentator.setClassifier( classifier2 );
//    segmentator.trainClassifier();
// // open new input image 
//    将新加入的图片和其对应的gt加入到实例中：
    ImagePlus input2 = IJ.openImage( "C:\\Documents and Settings\\Administrator\\桌面\\1.png" );
//    // open corresponding binary label image
    ImagePlus labels2 = IJ.openImage( "C:\\Documents and Settings\\Administrator\\桌面\\1_label.png" );
    
    int count=0;
    ImageProcessor ip = imp.getProcessor();
    if (imp.getType() == ImagePlus.GRAY16||imp.getType() == ImagePlus.GRAY8) {  
        int width = ip.getWidth();  
        int height = ip.getHeight();  
//        byte[] new_pixels = new byte[width * height];  
        // set each pixel value to whatever, between -128 and 127  
        for (int y=0; y<height; y++) {  
            for (int x=0; x<width; x++) {  
                // Editing pixel at x,y position  
                if (ip.getPixel(y, x)==0)
                	count=count+1;
            }  
        }  
        // update ImageProcessor to new array  
//        ip.setPixels(new_pixels);  
    }  
    System.out.println(count);
   
    
//    // for all slices in input2, add white pixels as labels for class 2 and 
//    // black pixels as labels for class 1
    int numSamples = count;
    segmentator.addRandomBalancedBinaryData( input2, labels2, "class 2", "class 1" , numSamples);
//    segmentator.addRandomBalancedBinaryData( input2, labels2, "class 2", "class 1" , 2*numSamples);
 // apply classifier to current training image and get label result 
    segmentator.trainClassifier();
 // (set parameter to true to get probabilities)
    //设置为true得到概率图，否则是分类图,0是线程数目
 ImagePlus result = segmentator.applyClassifier( imp, 0, false );
//save data into a ARFF file 保存数据
 segmentator.saveData( "C:\\Documents and Settings\\Administrator\\桌面\\1_png.arff" );
 
 //输出混淆矩阵       现在有点问题，先不用
// int[][] confused_maxtrix = segmentator.getTestConfusionMatrix(result, labels2, 1, 0);
// System.out.printf("\nconfused_maxtrix:\ncell:\t%d %d\n",confused_maxtrix[0][0],confused_maxtrix[0][1]);
// System.out.printf("background:%d %d",confused_maxtrix[1][0],confused_maxtrix[1][1]);
 
//load training data from ARFF file 导入数据
//segmentator.loadTrainingData( "my-traces-data.arff" );
 // get result (float image)
//result = segmentator.getClassifiedImage();
 
//save classifier into a file (.model)\
//String classifier_name="C:\\Documents and Settings\\Administrator\\桌面\\1_png_classifier.model";
//segmentator.saveClassifier( classifier_name );
// IJ.save(result, "C:\\Documents and Settings\\Administrator\\桌面\\1.jpg");
 //图片显示，销毁
// imp.show(); 
// result.show();
// try
// {
//  Thread.currentThread();
//  Thread.sleep(5000);//毫秒 
// }
// catch(Exception e){}
 imp.close();
 result.close();
 imp.flush();
 result.flush();


}
else
{
// WindowManager.addWindow(result.getWindow());
//open test image
// 将分类器用于测试图片
	ImagePlus testImage = IJ.openImage( "C:\\Documents and Settings\\Administrator\\桌面\\1.png" );
	testImage.show();
	WekaSegmentation segmentator2 = new WekaSegmentation( testImage );
	segmentator2.loadClassifier( classifier_name );

//get result (labels float image)
	ImagePlus result = segmentator2.applyClassifier( testImage, 0, false );
	result.show();
}

    
    
    
    
    
    
    

	}
}
