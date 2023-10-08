from prophecy.cb.server.base.ComponentBuilderBase import *
from pyspark.sql import *
from pyspark.sql.functions import *
from prophecy.cb.server.base.datatypes import SInt
from prophecy.cb.ui.uispec import *
from prophecy.cb.ui.UISpecUtil import *



class HuggingFaceLoad(ComponentSpec):
    name: str = "HuggingFaceLoad"
    category: str = "Transform"

    def optimizeCode(self) -> bool:
        return True

    @dataclass(frozen=True)
    class HuggingFaceLoadProperties(ComponentProperties):
        fileName: str = ""
        datasetType: str = ""
        columnName: str = ""



    def dialog(self) -> Dialog:
        InOutSchema = (PortSchemaTabs().importSchema())
        FileName = (ExpressionBox("File Name")
                    .bindPlaceholder(" ")
                    .bindProperty("fileName")
                    .withFrontEndLanguage())
        DatasetType = (ExpressionBox("Dataset Type")
                       .bindPlaceholder(" ")
                       .bindProperty("datasetType")
                       .withFrontEndLanguage())
        ColumnName = (ExpressionBox("Column Name")
                      .bindPlaceholder(" ")
                      .bindProperty("columnName")
                      .withFrontEndLanguage())

        return Dialog("word2vector").addElement(
            ColumnsLayout(gap="1rem", height="100%")
            .addColumn(InOutSchema, "2fr")
            .addColumn(
                StackLayout(height="100%")
                .addElement(FileName)
                .addElement(DatasetType)
                .addElement(ColumnName)
                , "2fr")
        )


    def validate(self, component: Component[HuggingFaceLoadProperties]) -> List[Diagnostic]:
        return []

    def onChange(self, oldState: Component[HuggingFaceLoadProperties], newState: Component[HuggingFaceLoadProperties]) -> Component[HuggingFaceLoadProperties]:
        return newState

    class HuggingFaceLoadCode(ComponentCode):

        def __init__(self, props):
            self.props: HuggingFaceLoad.HuggingFaceLoadProperties = props

        def apply(self, spark: SparkSession) -> DataFrame:
            from datasets import load_dataset
            from pyspark.sql import SparkSession
            from pyspark.ml.feature import Word2Vec

            a = load_dataset(self.props.datasetType, self.props.fileName)

            ds = []
            for i in a[self.props.columnName]["text"]:
                ds.append((i,))

            df = spark.createDataFrame(ds, ["text"])

            df1 = df.withColumn("word", split(col("text"), " "))

            return df1




