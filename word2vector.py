from prophecy.cb.server.base.ComponentBuilderBase import *
from pyspark.sql import *
from pyspark.sql.functions import *
from prophecy.cb.server.base.datatypes import SInt
from prophecy.cb.ui.uispec import *
from prophecy.cb.ui.UISpecUtil import *



class word2vector(ComponentSpec):
    name: str = "word2vector"
    category: str = "Transform"

    def optimizeCode(self) -> bool:
        return True

    @dataclass(frozen=True)
    class word2vectorProperties(ComponentProperties):
        Input_Col: str = ""
        Output_Col: str = ""
        Vector_Size: SInt = SInt("3")
        min_Count: SInt = SInt("0")
        num_Partitions: SInt = SInt("1")
        Step_Size: Optional[SFloat] = None
        max_Iter: SInt = SInt("1")
        Window_Size: SInt = SInt("5")
        Max_SentenceLength: SInt = SInt("1000")
        modelPath: str = ""


    def dialog(self) -> Dialog:
        InOutSchema = (PortSchemaTabs().importSchema())
        InputCol = ((SchemaColumnsDropdown("Input Column") \
                     .bindSchema("component.ports.inputs[0].schema")
                     .bindProperty("Input_Col")))
        OutputCol = (ExpressionBox("Output Column")
                     .bindPlaceholder(" ")
                     .bindProperty("Output_Col")
                     .withFrontEndLanguage())
        VectorSize = (ExpressionBox("Vector Size")
                      .bindPlaceholder("3")
                      .bindProperty("Vector_Size")
                      .withFrontEndLanguage())
        MinCount = (ExpressionBox("Min Count")
                    .bindPlaceholder("0")
                    .bindProperty("min_Count")
                    .withFrontEndLanguage())
        NumPartitions = (ExpressionBox("Num Partitions")
                         .bindPlaceholder("1")
                         .bindProperty("num_Partitions")
                         .withFrontEndLanguage())
        StepSize = (ExpressionBox("Step Size")
                    .bindPlaceholder("0.025")
                    .bindProperty("Step_Size")
                    .withFrontEndLanguage())
        MaxIter = (ExpressionBox("Max Iter")
                   .bindPlaceholder("1")
                   .bindProperty("max_Iter")
                   .withFrontEndLanguage())
        WindowSize = (ExpressionBox("Window Size")
                      .bindPlaceholder("5")
                      .bindProperty("Window_Size")
                      .withFrontEndLanguage())
        MaxSentenceLength = (ExpressionBox("Max Sentence Length")
                             .bindPlaceholder("1000")
                             .bindProperty("Max_SentenceLength")
                             .withFrontEndLanguage())
        ModelPath = (ExpressionBox("Model Path")
                     .bindPlaceholder("Eg: dbfs:/FileStore/TrainedModel")
                     .bindProperty("modelPath")
                     .withFrontEndLanguage())

        return Dialog("word2vector").addElement(
            ColumnsLayout(gap="1rem", height="100%")
            .addColumn(InOutSchema, "2fr")
            .addColumn(
                StackLayout(height="100%")
                .addElement(InputCol)
                .addElement(OutputCol)
                .addElement(VectorSize)
                .addElement(MinCount)
                .addElement(NumPartitions)
                .addElement(StepSize)
                .addElement(MaxIter)
                .addElement(WindowSize)
                .addElement(MaxSentenceLength)
                .addElement(ModelPath)
                , "2fr")
        )


    def validate(self, component: Component[word2vectorProperties]) -> List[Diagnostic]:
        return []

    def onChange(self, oldState: Component[word2vectorProperties], newState: Component[word2vectorProperties]) -> Component[word2vectorProperties]:
        return newState

    class word2vectorCode(ComponentCode):

        def __init__(self, props):
            self.props: word2vector.word2vectorProperties = props

        def apply(self, spark: SparkSession, in0: DataFrame) -> DataFrame:
            from pyspark.sql import SparkSession
            from pyspark.ml.feature import Word2Vec

            vec = Word2Vec(vectorSize=self.props.Vector_Size, minCount=self.props.min_Count, numPartitions=self.props.num_Partitions, stepSize = self.props.Step_Size, maxIter = self.props.max_Iter, inputCol=self.props.Input_Col, outputCol=self.props.Output_Col, windowSize = self.props.Window_Size, maxSentenceLength = self.props.Max_SentenceLength)

            model = vec.fit(in0)
            result = model.transform(in0)

            vecDf = model.getVectors()

            vecDf.write.parquet(self.props.modelPath)

            return vecDf
