package mySeg;
 
import hr.irb.fastRandomForest.FastRandomForest;
import ij.IJ;
import ij.ImagePlus;

import java.io.File;
import java.io.FilenameFilter;
import java.io.IOException;

import trainableSegmentation.WekaSegmentation;
import weka.classifiers.functions.SMO;
import weka.classifiers.trees.J48;


public class segment {

	enum classifiersEnum{SMO, J48, FRF};

	public static void main(String[] args) throws IOException{
		
		//特征选择,放到外面，只有train调用
		boolean[] enableFeatures = new boolean[]{
	            true,   /* Gaussian_blur */
	            true,   /* Sobel_filter */
	            true,   /* Hessian */
	            true,   /* Difference_of_gaussians */
	            false,   /* Membrane_projections */
	            false,  /* Variance */
	            false,  /* Mean */
	            false,  /* Minimum */
	            false,  /* Maximum */
	            false,  /* Median */
	            false,  /* Anisotropic_diffusion */
	            false,  /* Bilateral */
	            false,  /* Lipschitz */
	            false,  /* Kuwahara */
	            false,  /* Gabor */
	            false,  /* Derivatives */
	            false,  /* Laplacian */
	            false,  /* Structure */
	            false,  /* Entropy */
	            false   /* Neighbors */
		};
		// 数据集地址
		String dataset = "E:\\datasets\\second_edition\\training_datasets\\Fluo-N2DH-SIM+\\";
		
		String im1 = dataset + "01\\t032.tif";
		String lab1 = dataset + "01_GT\\SEG\\man_seg032.tif";
		String im2 = dataset + "01\\t064.tif";
		String lab2 = dataset + "01_GT\\SEG\\man_seg064.tif";
		//保存分类器和训练图片数据
		String savename = "t032t064_4-16_0.5";
		String classifier_name = dataset + "01_GT_pre\\" + savename + "_classifier.model";
		String data_name = dataset + "01_GT_pre\\" + savename + ".arff";
		
		boolean train = true;
	

	if(!train)
	{
		ImagePlus train_im1 = IJ.openImage(im1,1);
		ImagePlus train_lab1 = IJ.openImage(lab1);
		
		ImagePlus train_im2 = IJ.openImage(im2,1);
		ImagePlus train_lab2 = IJ.openImage(lab2);
		

		// 得到图片的宽和高
		int height = train_im1.getHeight();
		int width = train_im1.getWidth();
		System.out.printf("图片高度：%d\n图片宽度：%d\n\n", height, width);
		
		// 以下部分用于3d数据集的多slice
//		if(false){
//			int num_slices = image.getNSlices();		
//			System.out.printf("%d, %d\n", labels.getBitDepth(), labels.getType());		
//			System.out.printf("%d\n", labels.getCurrentSlice());
//		}
		
		// 创建weka分类器对象
		WekaSegmentation seg = new WekaSegmentation();
		seg.setTrainingImage(train_im1);
		
		// Classifier 选定快速随机森林，可更改
		classifiersEnum classfier =  classifiersEnum.FRF;
		switch (classfier){	
			case SMO:
				SMO classifier1 = new SMO();
				seg.setClassifier(classifier1);
				break;
			case J48:
				J48 classifier2 = new J48();
				seg.setClassifier(classifier2);
				break;				
			case FRF:
				FastRandomForest rf = new FastRandomForest();
				seg.setClassifier(rf);
				// 树的个数
				rf.setNumTrees(200);        
				// Number of features per tree
				rf.setNumFeatures(2);
				// Seed
				rf.setSeed(0);// (new java.util.Random()).nextInt();
				break;
		}
		// 设置参数 主要是sigma
		// membrane patch size 没有用到这个
//		seg.setMembraneThickness(1);
//		seg.setMembranePatchSize(19);
		seg.setMaximumSigma(16.0f);
		seg.setMinimumSigma(4.0f);     //MinimumSigma设为0和设为1是一样的
		
		//下面这部分用于获得前景点个数, 并将前景都变为白色		
		int count1 = train_lab1.Convert2Binary();
		int count2 = train_lab2.Convert2Binary();
		// 显示一下图片，以验证 Convert2Binary() 没有出错,显示5秒后自动关闭
//        train_lab1.show();
//        train_lab2.show();
        
		// 统计2张图片前景点的个数
	    System.out.printf("%s共有前景像素点 %d 个\n", im1, count1);
	    System.out.printf("%s共有前景像素点 %d 个\n", im2, count2);
	    
	    //选取部分的前景和数量相等的背景
		int nSamplesToUse1 = (int) (count1*0.5);
		int nSamplesToUse2 = (int) (count2*0.5);
		
//###########################################################################//	
		
		// 选择是否从样本中进行训练
		if(true){		
			// Enable features in the segmentator
			seg.setEnabledFeatures( enableFeatures );
			// Add balanced labels  这一步中已经加入了特征提取，即add数据+特征提取
			seg.addRandomBalancedBinaryData(train_im1, train_lab1, "class 2", "class 1", nSamplesToUse1);
			seg.addRandomBalancedBinaryData(train_im2, train_lab2, "class 2", "class 1", nSamplesToUse2);
			// Train classifier
			seg.trainClassifier();
			
			// 选择是否保存训练数据和分类器
			if(true){
				seg.saveData( data_name );	
				seg.saveClassifier( classifier_name );
			}
			
		}
		else	// 从外部载入训练数据和分类器
		{		
			seg.loadTrainingData( data_name ); 
			seg.loadClassifier( classifier_name );
		}
		
//###########################################################################//	
		int[][] cm;
		ImagePlus classif_result = new ImagePlus();

		
//###########################################################################//	
		
		// 选择是否对训练图片进行测试
		if(true){
			
			seg.applyClassifier( false );
			classif_result = seg.getClassifiedImage();
			train_lab1.show();
			cm = seg.getConfusionMatrix( classif_result,train_lab1, 0, 1);
		}
		else{ //载入新的测试图片进行测试

			ImagePlus test = train_im2;
			classif_result = seg.applyClassifier(test, 0, false);
			ImagePlus label_test = train_lab2;		
//			label_test.Convert2Binary();
			train_lab2.show();
	        
	        cm = seg.getConfusionMatrix( classif_result, label_test, 0, 1);
			
		}
		
		// 展示分割结果，混淆矩阵
		classif_result.show();

		System.out.printf("===ConfusionMatrix===\n%d %d\n%d %d",cm[0][0],cm[0][1],cm[1][0],cm[1][1]);
		//销毁图像
		train_im1.close();
//		train_lab1.close();
		train_im2.close();
//		train_lab2.close();

	}
	else
	{
	///////////////////////////////////////////////////////////////////////////////////////////////
	///////////////////////////////////////////////////////////////////////////////////////////////
		// 读入测试图片文件夹
		String dataset2 = dataset.replace("training", "competition");
		
		File imdir = new File( dataset2 + "01" );
		if (!imdir.isDirectory())
			throw new RuntimeException("测试文件目录错误！");		
		
		// 找到后缀为tif的图片文件
		String[] im_name = imdir.list(new FilenameFilter(){
            public boolean accept(File dir,String name){
                return name.endsWith(".tif");
            }
        });
		
//		System.out.println(im_name[10]);
		// 建立保存目录
		File save_dir=new File( dataset2 + "01_4-16_seg" );    
		if(!save_dir.exists()) //文件夹不存在
		{    
		    save_dir.mkdir();    
		}    
		
		// 只对GT里面含有的图片进行处理
//		File GT_dir=new File( "E:\\datasets\\first editon\\training datasets\\N2DL-HeLa\\01_GT\\SEG" ); 
//		String[] gt_name = GT_dir.list(new FilenameFilter(){
//            public boolean accept(File dir,String name){
//                return name.endsWith(".tif");
//            }
//        });
		
		ImagePlus vitualtrain = IJ.openImage(dataset + "01\\t000.tif");
		WekaSegmentation seg_test = new WekaSegmentation( vitualtrain );
		seg_test.loadClassifier( classifier_name );
		
		//im_name.length
		for(int n=0; n<im_name.length; n++){
//        //循环处理所有图片
			String im_full_name = imdir.getAbsolutePath() + "\\" + im_name[n];
			System.out.println("====================");
			System.out.printf("reading image:\n%s\n",im_full_name);
			
			ImagePlus test = IJ.openImage(im_full_name);
			// 载入分类器,得到分类结果
//			seg_test.setTrainingImage(test);
//			WekaSegmentation seg_test = new WekaSegmentation(test);
			ImagePlus result = seg_test.applyClassifier(test, 0, false);
//			seg_test.applyClassifier(false);
//			ImagePlus result = seg_test.getClassifiedImage(); 

			//保存分割图到目录中
			String save_name = save_dir.getAbsolutePath() + "\\" + im_name[n];
			System.out.printf("saving image:\n%s\n",save_name);
			// 转换为unit16, 并保存为 tif 格式
//			new ImageConverter( result ).convertToGray16();
			IJ.save(result, save_name);
			System.out.printf("done!\n\n");
			test.flush();
			result.flush();	
		}
		
	}  //测试部分结束
	}  //main函数结束
}
