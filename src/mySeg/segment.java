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
		
		//����ѡ��,�ŵ����棬ֻ��train����
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
		// ���ݼ���ַ
		String dataset = "E:\\datasets\\second_edition\\training_datasets\\Fluo-N2DH-SIM+\\";
		
		String im1 = dataset + "01\\t032.tif";
		String lab1 = dataset + "01_GT\\SEG\\man_seg032.tif";
		String im2 = dataset + "01\\t064.tif";
		String lab2 = dataset + "01_GT\\SEG\\man_seg064.tif";
		//�����������ѵ��ͼƬ����
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
		

		// �õ�ͼƬ�Ŀ�͸�
		int height = train_im1.getHeight();
		int width = train_im1.getWidth();
		System.out.printf("ͼƬ�߶ȣ�%d\nͼƬ��ȣ�%d\n\n", height, width);
		
		// ���²�������3d���ݼ��Ķ�slice
//		if(false){
//			int num_slices = image.getNSlices();		
//			System.out.printf("%d, %d\n", labels.getBitDepth(), labels.getType());		
//			System.out.printf("%d\n", labels.getCurrentSlice());
//		}
		
		// ����weka����������
		WekaSegmentation seg = new WekaSegmentation();
		seg.setTrainingImage(train_im1);
		
		// Classifier ѡ���������ɭ�֣��ɸ���
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
				// ���ĸ���
				rf.setNumTrees(200);        
				// Number of features per tree
				rf.setNumFeatures(2);
				// Seed
				rf.setSeed(0);// (new java.util.Random()).nextInt();
				break;
		}
		// ���ò��� ��Ҫ��sigma
		// membrane patch size û���õ����
//		seg.setMembraneThickness(1);
//		seg.setMembranePatchSize(19);
		seg.setMaximumSigma(16.0f);
		seg.setMinimumSigma(4.0f);     //MinimumSigma��Ϊ0����Ϊ1��һ����
		
		//�����ⲿ�����ڻ��ǰ�������, ����ǰ������Ϊ��ɫ		
		int count1 = train_lab1.Convert2Binary();
		int count2 = train_lab2.Convert2Binary();
		// ��ʾһ��ͼƬ������֤ Convert2Binary() û�г���,��ʾ5����Զ��ر�
//        train_lab1.show();
//        train_lab2.show();
        
		// ͳ��2��ͼƬǰ����ĸ���
	    System.out.printf("%s����ǰ�����ص� %d ��\n", im1, count1);
	    System.out.printf("%s����ǰ�����ص� %d ��\n", im2, count2);
	    
	    //ѡȡ���ֵ�ǰ����������ȵı���
		int nSamplesToUse1 = (int) (count1*0.5);
		int nSamplesToUse2 = (int) (count2*0.5);
		
//###########################################################################//	
		
		// ѡ���Ƿ�������н���ѵ��
		if(true){		
			// Enable features in the segmentator
			seg.setEnabledFeatures( enableFeatures );
			// Add balanced labels  ��һ�����Ѿ�������������ȡ����add����+������ȡ
			seg.addRandomBalancedBinaryData(train_im1, train_lab1, "class 2", "class 1", nSamplesToUse1);
			seg.addRandomBalancedBinaryData(train_im2, train_lab2, "class 2", "class 1", nSamplesToUse2);
			// Train classifier
			seg.trainClassifier();
			
			// ѡ���Ƿ񱣴�ѵ�����ݺͷ�����
			if(true){
				seg.saveData( data_name );	
				seg.saveClassifier( classifier_name );
			}
			
		}
		else	// ���ⲿ����ѵ�����ݺͷ�����
		{		
			seg.loadTrainingData( data_name ); 
			seg.loadClassifier( classifier_name );
		}
		
//###########################################################################//	
		int[][] cm;
		ImagePlus classif_result = new ImagePlus();

		
//###########################################################################//	
		
		// ѡ���Ƿ��ѵ��ͼƬ���в���
		if(true){
			
			seg.applyClassifier( false );
			classif_result = seg.getClassifiedImage();
			train_lab1.show();
			cm = seg.getConfusionMatrix( classif_result,train_lab1, 0, 1);
		}
		else{ //�����µĲ���ͼƬ���в���

			ImagePlus test = train_im2;
			classif_result = seg.applyClassifier(test, 0, false);
			ImagePlus label_test = train_lab2;		
//			label_test.Convert2Binary();
			train_lab2.show();
	        
	        cm = seg.getConfusionMatrix( classif_result, label_test, 0, 1);
			
		}
		
		// չʾ�ָ�������������
		classif_result.show();

		System.out.printf("===ConfusionMatrix===\n%d %d\n%d %d",cm[0][0],cm[0][1],cm[1][0],cm[1][1]);
		//����ͼ��
		train_im1.close();
//		train_lab1.close();
		train_im2.close();
//		train_lab2.close();

	}
	else
	{
	///////////////////////////////////////////////////////////////////////////////////////////////
	///////////////////////////////////////////////////////////////////////////////////////////////
		// �������ͼƬ�ļ���
		String dataset2 = dataset.replace("training", "competition");
		
		File imdir = new File( dataset2 + "01" );
		if (!imdir.isDirectory())
			throw new RuntimeException("�����ļ�Ŀ¼����");		
		
		// �ҵ���׺Ϊtif��ͼƬ�ļ�
		String[] im_name = imdir.list(new FilenameFilter(){
            public boolean accept(File dir,String name){
                return name.endsWith(".tif");
            }
        });
		
//		System.out.println(im_name[10]);
		// ��������Ŀ¼
		File save_dir=new File( dataset2 + "01_4-16_seg" );    
		if(!save_dir.exists()) //�ļ��в�����
		{    
		    save_dir.mkdir();    
		}    
		
		// ֻ��GT���溬�е�ͼƬ���д���
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
//        //ѭ����������ͼƬ
			String im_full_name = imdir.getAbsolutePath() + "\\" + im_name[n];
			System.out.println("====================");
			System.out.printf("reading image:\n%s\n",im_full_name);
			
			ImagePlus test = IJ.openImage(im_full_name);
			// ���������,�õ�������
//			seg_test.setTrainingImage(test);
//			WekaSegmentation seg_test = new WekaSegmentation(test);
			ImagePlus result = seg_test.applyClassifier(test, 0, false);
//			seg_test.applyClassifier(false);
//			ImagePlus result = seg_test.getClassifiedImage(); 

			//����ָ�ͼ��Ŀ¼��
			String save_name = save_dir.getAbsolutePath() + "\\" + im_name[n];
			System.out.printf("saving image:\n%s\n",save_name);
			// ת��Ϊunit16, ������Ϊ tif ��ʽ
//			new ImageConverter( result ).convertToGray16();
			IJ.save(result, save_name);
			System.out.printf("done!\n\n");
			test.flush();
			result.flush();	
		}
		
	}  //���Բ��ֽ���
	}  //main��������
}
