package mySeg;

import ij.process.ImageConverter;
import ij.process.ImageProcessor;


// 这个类继承 ij.ImagePlus 类，但无法运行，因此将这个Convert2Binary函数直接写到 ij.ImagePlus 的首部
public class Label_Convert2Binary extends ij.ImagePlus{
	
	public int Convert2Binary(){
		int count=0;
		new ImageConverter( this ).convertToGray8();
	    ImageProcessor ip = this.getProcessor();
	    // 图像格式gray8为0，gray16为1，gray32为2
//	    if (image.getType() < 3) {  
        byte[] new_pixels = new byte[width * height];  
        // set each pixel value to whatever, between -128 and 127  
        for (int y=0; y<height; y++) {  
            for (int x=0; x<width; x++) {  
                // Editing pixel at x,y position  
                if (ip.getPixel(x, y)!=0){
                	new_pixels[y * width + x] = (byte) 255;
                	count=count+1;
                }
            }  
        }  
        ip.setPixels(new_pixels);
        
		return count;		
	}
	

}
