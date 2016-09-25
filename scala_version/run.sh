ROOT=$(cd "$(dirname $0)/.."; pwd)

CLASS_PATH=$ROOT/scala_version/target/scala-2.11/classes/:\
$HOME/.ivy2/cache/org.scala-lang/scala-library/jars/scala-library-2.11.8.jar:\
$HOME/.ivy2/cache/nu.pattern/opencv/jars/opencv-2.4.9-7.jar:\
$HOME/.ivy2/cache/commons-io/commons-io/jars/commons-io-2.5.jar


DATA_PATH=$ROOT/datas
MODEL_PATH=$ROOT/model


java -Xmx4G -cp $CLASS_PATH \
	PCANetScala  $DATA_PATH $MODEL_PATH