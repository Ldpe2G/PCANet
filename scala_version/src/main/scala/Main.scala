  import org.opencv.core.Core
import org.opencv.core.CvType
import org.opencv.core.Mat
import scala.io.Source
import java.io.File
import org.apache.commons.io.FileUtils
import org.opencv.highgui.Highgui
import org.opencv.core.Size
import org.opencv.imgproc.Imgproc
import Utils._
import org.opencv.ml.CvSVMParams
import org.opencv.ml.CvSVM
import org.opencv.core.TermCriteria

object Main {


  def main(args: Array[String]): Unit = {
     
    
      System.loadLibrary( Core.NATIVE_LIBRARY_NAME)
      
      val DIR_NUM = 7
      //input image size height: 60, width: 48
      // 路径根据自己的情况来修改即可
      val train_dir = Array(
        ".\\datas\\train\\1\\1_",
        ".\\datas\\train\\2\\2_",
        ".\\datas\\train\\3\\3_",
        ".\\datas\\train\\4\\4_",
        ".\\datas\\train\\5\\5_",
        ".\\datas\\train\\6\\6_",
        ".\\datas\\train\\7\\7_"
      )
    
      val test_dir = Array(
        "datas\\test\\1\\1_",
        "datas\\test\\2\\2_",
        "datas\\test\\3\\3_",
        "datas\\test\\4\\4_",
        "datas\\test\\5\\5_",
        "datas\\test\\6\\6_",
        "datas\\test\\7\\7_"
      )
      
      val NumFilters = List(8, 8)
      val blockSize = List(12, 10)
  
      val pcaNet = PCANet(
        2,
        7,
        NumFilters,
        blockSize,
        0.5   
      )
  
      val train_num = 40
      val NUM = DIR_NUM * train_num
      
      
      val (inImgs, labels) = ((List[Mat](), List[Float]()) /: (1 to train_num)){
        case ((imgs, labs), idx) =>
            val (tmp_imgs, tmp_labels) = ((imgs, labs) /: (0 until DIR_NUM)){
              case ((ims, las), dir) =>
                val img = Highgui.imread(s"${train_dir(dir)}$idx.jpg", 1)
                val changed = new Mat(img.rows(), img.cols(), CvType.CV_64F)
                img.convertTo(changed, CvType.CV_64F, 1.0 / 255)        
                (ims :+ changed, las :+ (dir + 1).toFloat)
            }
            (tmp_imgs, tmp_labels)              
      }
        
      val train_result = PCANet_train(inImgs, pcaNet, true)
      
      println("\n ====== Training Linear SVM Classifier ======= ")
      
      val new_labels = (Array[Float]() /: train_result.feature_idx){
        case (array, idx) =>
          array :+ labels(idx)
      }
      
      val labelsMat = new Mat(NUM, 1, CvType.CV_32FC1)
      labelsMat.put(0, 0, new_labels)
      
      val features = new Mat
      train_result.Features.convertTo(features, CvType.CV_32F)
      
      val params = new CvSVMParams  
      params.set_svm_type(CvSVM.C_SVC)
      params.set_C(1)
      params.set_kernel_type(CvSVM.LINEAR)
      params.set_term_crit(new TermCriteria(TermCriteria.EPS, TermCriteria.MAX_ITER, 1e-6))
      //终止准则函数：当迭代次数达到最大值时终止  
  
    //训练SVM  
    //建立一个SVM类的实例  
     val SVM = new CvSVM
      
     var e1 = Core.getTickCount
       
     SVM.train(features, labelsMat, new Mat, new Mat, params)  

     var e2 = Core.getTickCount
     var time = (e2 - e1) / Core.getTickFrequency
     println(s" svm training complete, time usage: $time")
      
     SVM.save(".\\model\\all_age_svm.xml")
     
     
     println("\n ====== PCANet Testing ======= ")
      
     val (testImgs, testLabels) = ((List[Mat](), List[Float]()) /: (41 to 64)){
        case ((imgs, labs), idx) =>
            val (tmp_imgs, tmp_labels) = ((imgs, labs) /: (0 until DIR_NUM)){
              case ((ims, las), dir) =>
                val img = Highgui.imread(s"${test_dir(dir)}$idx.jpg", 1)
                val changed = new Mat(img.rows(), img.cols(), CvType.CV_64F)
                img.convertTo(changed, CvType.CV_64F, 1.0 / 255)        
                (ims :+ changed, las :+ (dir + 1).toFloat)
            }
            (tmp_imgs, tmp_labels)              
      }
        
      val testNum = 24
      val all = DIR_NUM * testNum
      e1 = Core.getTickCount

      val (corrs, correct) = (((1 to DIR_NUM).map(x => 0).toArray, 0) /: testImgs.zip(testLabels)){
        case ((tmp_corrs, tmp_correct), (img, label)) =>
            val out = PCA_output(List(img), List(0), pcaNet.PatchSize, pcaNet.NumFilters(0), train_result.Filters(0))
    
            val out2 = PCA_output(out.OutImg, out.OutImgIdx ++ (1 until pcaNet.NumFilters(1)), pcaNet.PatchSize, 
                    pcaNet.NumFilters(1), train_result.Filters(1)) 
            val hashing_r = HashingHist(pcaNet, out2.OutImgIdx, out2.OutImg) 
            val fea = new Mat
            hashing_r.Features.convertTo(fea, CvType.CV_32F)
            val la = SVM.predict(fea)
             if (la == label) {
                tmp_corrs(la.toInt - 1) = tmp_corrs(la.toInt - 1) + 1
                (tmp_corrs, tmp_correct + 1)
            } else (tmp_corrs, tmp_correct)
      }
      
       e2 = Core.getTickCount
       time = (e2 - e1) / Core.getTickFrequency
       println(s" test time usage: $time")
      
      println(s"all precise: ${correct.toFloat / all}")
      for(i <- 0 until DIR_NUM)
        println(s"person${i+1} precise: ${corrs(i).toFloat / testNum}")
        
      println(s"test images num for each class: $testNum")
  }
  
}