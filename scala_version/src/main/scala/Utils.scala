import org.opencv.core.Mat
import org.opencv.core.Core
import java.util.ArrayList
import org.opencv.imgproc.Imgproc
import org.opencv.core.Scalar
import scala.util.Random

object Utils {
  
    
   case class PCA_Out_Result(OutImg: List[Mat], OutImgIdx: List[Int])
   case class Hashing_Result(Features: Mat, BlkIdx: List[Int])
   case class PCANet(NumStages: Int, PatchSize: Int, 
                     NumFilters: List[Int], HistBlockSize: List[Int],
                     BlkOverLapRatio: Double)
            
   case class PCA_Train_Result(Features: Mat, feature_idx: List[Int],
                    Filters: List[Mat], BlkIdx: List[Int])
       
     
    
    val ROW_DIM = 0
    val COL_DIM = 1
    
    def im2colstep(InImg: Mat, blockSize: List[Int], stepSize: List[Int]) = {
        val r_row = blockSize(ROW_DIM) * blockSize(COL_DIM)
        val row_diff = InImg.rows - blockSize(ROW_DIM)
        val col_diff = InImg.cols - blockSize(COL_DIM)
        val r_col = (row_diff / stepSize(ROW_DIM) + 1) * (col_diff / stepSize(COL_DIM) + 1)
        val OutBlocks = new Mat(r_col, r_row, InImg.depth())
        
        var blocknum = 0

        val size = (InImg.total * InImg.channels).toInt
        val buff = new Array[Double](size)
        
        InImg.get(0, 0, buff)
      
        val size2 = r_col * r_row
      
        val Obuff = new Array[Double](size2)
        val col = InImg.cols

        for(j <- 0 to col_diff by stepSize(COL_DIM)){
          for(i <- 0 to row_diff by stepSize(ROW_DIM)){
            
            for(m <- 0 until blockSize(ROW_DIM)){
              for(l <- 0 until blockSize(COL_DIM))
                Obuff(blocknum * r_row + blockSize(ROW_DIM) * l + m) = buff((i + m) * col + j + l)
            }
            blocknum = blocknum + 1
          }
        }
        
        OutBlocks.put(0, 0, Obuff: _*)
        OutBlocks
    }
    
    def im2col_general(InImg: Mat, blockSize: List[Int], stepSize: List[Int]) = {
        val channels = InImg.channels
    
        val layers = new ArrayList[Mat]()
      
        if(channels > 1)
            Core.split(InImg, layers)
        else
            layers.add(InImg)
      
        val AllBlocks = new Mat
        val src = new ArrayList[Mat]()
      
        val size = layers.size
      
        for(i <- 0 until size)
            src.add(im2colstep(layers.get(i), blockSize, stepSize))  
                  
        Core.hconcat(src, AllBlocks)
      
        AllBlocks.t
    }
 
    
    def PCA_output(InImg: List[Mat], InImgIdx: List[Int], PatchSize: Int, NumFilters: Int, Filters: Mat) = {
        val img_length = InImg.size
        val mag = (PatchSize - 1) / 2
        
        val (blockSize, stepSize) = ((List[Int](), List[Int]()) /: (1 to 2)){(acc, elem) => 
          (acc._1 :+ PatchSize, acc._2 :+ 1)
        }
        
        val (outImgs, outImgIdx) = ((List[Mat](), List[Int]()) /: InImg.zip(InImgIdx)){
          case ((outs, outidx), (image, index)) =>
            val img = new Mat
            Imgproc.copyMakeBorder(image, img, mag, mag, mag, mag, Imgproc.BORDER_ISOLATED);
            val temp = im2col_general(img, blockSize, stepSize)
            val mean = new Mat
            Core.reduce(temp, mean, 0, Core.REDUCE_AVG)
            val temp3 = new Mat
            temp3.create(0, temp.cols(), temp.`type`())
            val temp2 = new Mat
            for(j <- 0 until temp.rows()){
                Core.subtract(temp.row(j), mean.row(0), temp2)
                temp3.push_back(temp2.row(0))
            } 
            val OutImgIdx = outidx :+ index
            val fake3 = new Mat
            val OutImgs = (outs /: (0 until NumFilters)){
              case (outImgs, filterIdx) =>
                Core.gemm(Filters.row(filterIdx), temp3, 1, fake3, 0, temp);  
            
                outImgs :+ temp.reshape(0, image.cols()).t
      
            }
            (OutImgs, OutImgIdx)  
        }
        
        PCA_Out_Result(outImgs, outImgIdx)
    }
    
    def PCA_FilterBank(InImg: List[Mat], PatchSize: Int, NumFilters: Int) = {
        val channels = InImg(0).channels
        val InImg_Size = InImg.size

        //val randIdx = getRandom(InImg_Size)

        val size = channels * PatchSize * PatchSize
        val img_depth = InImg(0).depth
        
        val (blockSize, stepSize) = ((List[Int](), List[Int]()) /: (0 until 2)){(acc, elem) =>
            (acc._1 :+ PatchSize, acc._2 :+ 1)  
        }
        var cols = 0
        val Rx = (Mat.zeros(size, size, img_depth) /: (0 until InImg_Size)){
          case (sum, idx) =>
              
              val temp = im2col_general(InImg(idx), blockSize, stepSize)
              cols = temp.cols
              val mean = new Mat
              Core.reduce(temp, mean, 0, Core.REDUCE_AVG)
              val temp3 = new Mat
              temp3.create(0, temp.cols, temp.`type`)
             
              val temp2 = new Mat
              for(i <- 0 until temp.rows){
                   Core.subtract(temp.row(i), mean.row(0), temp2)
                   temp3.push_back(temp2.row(0))
              }
              Core.gemm(temp3, temp3.t, 1, new Mat, 0, temp)  
              Core.add(sum, temp, sum)
              sum
        }

        //Core.multiply(Rx, new Scalar(1.0 / (InImg_Size * Rx.cols)), Rx)
        //val Rxx = cols
        //Core.divide(Rx, new Scalar((InImg_Size * Rx.cols).toDouble), Rx)
        Core.multiply(Rx, new Scalar(1.0 / (InImg_Size * cols).toDouble), Rx)
        
        val eValuesMat = new Mat
        val eVectorsMat = new Mat
        
        Core.eigen(Rx, true, eValuesMat, eVectorsMat)
        
        val Filters = new Mat(0, Rx.cols, Rx.depth)
        
        for(i <- 0 until NumFilters)
            Filters.push_back(eVectorsMat.row(i))
            
       Filters
    }
    
    def PCANet_train(InImg: List[Mat], PcaNet: PCANet, is_extract_feature: Boolean): PCA_Train_Result = {
        assert(PcaNet.NumFilters.size == PcaNet.NumStages)
          
        val img_length = InImg.size

        val OutImgIdx = (0 until img_length).toList
        
        var e1, eo1, eo2, eb1, eb2 = Core.getTickCount
        
        val (out_result, filters) = ((PCA_Out_Result(InImg, OutImgIdx), List[Mat]()) /: (0 until PcaNet.NumStages)){
          case ((result, filter), stage) =>
              eb1 = Core.getTickCount
              println(s" Computing PCA filter bank and its outputs at stage $stage ...")
              val fts = PCA_FilterBank(result.OutImg, PcaNet.PatchSize, PcaNet.NumFilters(stage))
              eb2 = Core.getTickCount
              println(s" stage $stage PCA_FilterBank time: ${(eb2 - eb1)/ Core.getTickFrequency}")

              eo1 = Core.getTickCount
              var temp = result
              if(stage != PcaNet.NumStages - 1){
                temp = PCA_output(result.OutImg, result.OutImgIdx, PcaNet.PatchSize, 
                                  PcaNet.NumFilters(stage), fts)
              }
              eo2 = Core.getTickCount
              println(s" stage $stage output time: ${(eo2 - eo1) / Core.getTickFrequency}")
              (temp, filter :+ fts)
         }
         var e2 = Core.getTickCount
         var time = (e2 - e1) / Core.getTickFrequency
         println(s"\n totle FilterBank time: $time")
         
         
         val end = PcaNet.NumStages - 1
         val outIdx_length = out_result.OutImgIdx.size
         
          val (features, features_idx) = if(is_extract_feature){
                  
            e1 = Core.getTickCount
            val (fea_list, fea_idx) = ((List[Mat](), List[Int]()) /: (0 until img_length)){
              case ((fea_l, fea_i), img_idx) =>
                  val subInImg = out_result.OutImg.drop(img_idx * PcaNet.NumFilters(end))
                                                  .take(PcaNet.NumFilters(end))
                  val subIdx = (0 until PcaNet.NumFilters(end)).toList
                  
                  val temp = PCA_output(subInImg, subIdx, PcaNet.PatchSize, PcaNet.NumFilters(end), filters(end))
                  
                  val hashing_r = HashingHist(PcaNet, temp.OutImgIdx, temp.OutImg)

                  (fea_l :+ hashing_r.Features, fea_i :+ out_result.OutImgIdx(img_idx))
                  
            } 
            e2 = Core.getTickCount
            time = (e2 - e1) / Core.getTickFrequency
            println(s"\n hasing time: $time")
             
            
            val size = fea_list.size
            val Features =  if(size > 0){
                val temp = new Mat
                temp.create(0, fea_list(0).cols, fea_list(0).`type`)
                for(i <- 0 until size){
                  temp.push_back(fea_list(i))
              }
              temp
            }else new Mat
 
            (Features, fea_idx)
          } else (new Mat, List[Int]())

         
         PCA_Train_Result(features, features_idx, filters, List[Int]())
    }
    
      
  def round(r: Double) = {  
      if(r > 0.0)  Math.floor(r + 0.5) else Math.ceil(r - 0.5)  
  }  
  
  
  def Heaviside(X: Mat) = {
      val row = X.rows
      val col = X.cols
      val types = X.`type`
      
      val H = Mat.zeros(row, col, types)
      
      val size = X.total.toInt * X.channels
      
      val XBuff = new Array[Double](size)
      X.get(0, 0, XBuff)
      
      
      val HBuff = new Array[Double](size)
      for(i <- 0  until row){
        val i_col = i * col
        for(j <- 0 until col){
          if(XBuff(i_col + j) > 0) 
            HBuff(i_col + j) = 1
          else 
            HBuff(i_col + j) = 0
        }
      }
     
      H.put(0, 0, HBuff: _*)
      H
  }
  
  def Hist(mat: Mat, Range: Int) = {
    val temp = mat.t
    val row = temp.rows
    val col = temp.cols
    val types = temp.`type`
    val Hist = Mat.zeros(row, Range + 1, types)

    val size = temp.total.toInt
    val tBuff = new Array[Double](size)
    temp.get(0, 0, tBuff)
    
    val size2 = (Range + 1) * row
    val hBuff = new Array[Double](size2)
    for(i <- 0 until size2) hBuff(i) = 0
    
    var tt: Int = 0
    var tt2: Int = 0
    for(i <- 0 until row){
      tt = i*(Range + 1)
      for(j <- 0 until col){
        tt2 = tt + tBuff(i*col + j).toInt
        hBuff(tt2) = hBuff(tt2) + 1
      }
    }
     
    Hist.put(0, 0, hBuff: _*)
    
    Hist.t

  }
  
  def bsxfun_times(BHist: Mat, NumFilters: Int) = {
      val row = BHist.rows
      val col = BHist.cols
      
      val size = BHist.total.toInt * BHist.channels
      
      val buff = new Array[Double](size)
      BHist.get(0, 0, buff)
    
      val p = Math.pow(2.0, NumFilters)

      val sum = (1 to col).map(x => 0).toArray
      val sum2 = buff.toList.grouped(col)
                 .map(_.zip(sum.toList).map(tu => tu._1 + tu._2))
                 .reduce((l, r) => l.zip(r).map(x => x._1 + x._2))
                 .map(p / _)
      /*
      val sum = new Array[Double](col)
      for(i <- 0 until col)
        sum(i) = 0
    
      for(i <- 0 until row)
        for(j <- 0 until col)
          sum(j) =  sum(j) + buff(i * col + j)
    */
      
      
      //val sum2 = sum.map(p / _)
      
      val temp: Array[Double] = buff.toList.grouped(col)
                                .map(_.zip(sum2).map(tu => tu._1 * tu._2))
                                .flatten.toArray
      //val temp = new Array[Double](size)
      
     /*for(i <- 0 until row)
        for(j <- 0 until col){
          temp[i * col + j] = buff[i * col + j] * sum(j)
        }
    */
      BHist.put(0, 0, temp: _*)
      BHist
  }
  
  def HashingHist(PcaNet: PCANet, ImgIdx: List[Int], Imgs: List[Mat]): Hashing_Result = {
      val length = Imgs.size
      val NumFilters = PcaNet.NumFilters(PcaNet.NumStages - 1)
      val NumImgin0 = length / NumFilters
      
      val tempMat = Imgs(0)
      val row = tempMat.rows
      val col = tempMat.cols
      val types = tempMat.`type`
      
      val map_weights = new Array[Double](NumFilters)

      for(i <- NumFilters - 1 to 0 by -1)
        map_weights(NumFilters - 1 - i) = Math.pow(2.0, i.toDouble)

      val rate = 1 - PcaNet.BlkOverLapRatio
      val Ro_BlockSize = (List[Int]() /: (0 until PcaNet.HistBlockSize.length)){(acc, elem) =>
            acc :+ round(PcaNet.HistBlockSize(elem) * rate).toInt
      }
      val BHist = new Mat

      val ImgIdx_length = ImgIdx.size
      val new_idx = new Array[Int](ImgIdx_length)
      for(i <- 0 until ImgIdx_length)
          new_idx(ImgIdx(i)) = i

      
      val matList = new ArrayList[Mat]

      var T = new Mat
      var scalar = new Scalar(0)
      var temp = new Mat
      
      for(i <- 0 until NumImgin0){
        T = Mat.zeros(row, col, types) 

        for(j <- 0 until NumFilters){
          temp = Heaviside(Imgs(NumFilters * new_idx(i) + j))
        
          scalar = new Scalar(map_weights(j))
          Core.multiply(temp, scalar, temp)
        
          Core.add(T, temp, T)
        }
      
        temp = im2col_general(T, PcaNet.HistBlockSize, Ro_BlockSize)
        temp = Hist(temp, Math.pow(2.0, NumFilters).toInt - 1)
      
        temp = bsxfun_times(temp, NumFilters)
      
        matList.add(temp)
      }
      
      Core.hconcat(matList, BHist)
    
      val rows = BHist.rows
      val cols = BHist.cols

      val Features = new Mat
      Features.create(1, rows * cols, BHist.`type`)

      val size = rows * cols
      val XBuff = new Array[Double](size)
      BHist.get(0, 0, XBuff)

      val HBuff = new Array[Double](size)
      
      for(i <- 0 until rows){
        for(j <- 0 until cols){  
          HBuff(j * rows + i) = XBuff(i * cols + j)    
        }
      }
      Features.put(0, 0, HBuff: _*)
      
      Hashing_Result(Features, List[Int]())
  }
    
    def getRandom(size: Int) = Random.shuffle((0 until size).toList)     

}

