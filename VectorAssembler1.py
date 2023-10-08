from prophecy.cb.server.base.ComponentBuilderBase import *
from pyspark.sql import *
from pyspark.sql.functions import *
from prophecy.cb.server.base.datatypes import SInt
from prophecy.cb.ui.uispec import *
from prophecy.cb.ui.UISpecUtil import *


class VectorAssembler1(ComponentSpec):
    name: str = "VectorAssembler1"
    category: str = "Transform"

    def optimizeCode(self) -> bool:
        return True

    @dataclass(frozen=True)
    class VectorAssembler1Properties(ComponentProperties):
        sourcePath: str = ""
        selectColumns: str = ""
        targetColumn: str = ""


    def dialog(self) -> Dialog:
        return Dialog("Linear Regression").addElement(
            ColumnsLayout(gap="1rem", height="100%")
            .addColumn(PortSchemaTabs().importSchema(), "2fr")
            .addElement((TextBox("Source Path", placeholder=" ")
                         .bindProperty("sourcePath")))
            .addElement((TextBox("Select Columns", placeholder=" ")
                         .bindProperty("selectColumns")))
            .addElement((TextBox("Target Column", placeholder=" ")
                         .bindProperty("targetColumn")))
        )

    def validate(self, component: Component[VectorAssembler1Properties]) -> List[Diagnostic]:
        return []

    def onChange(self, oldState: Component[VectorAssembler1Properties], newState: Component[VectorAssembler1Properties]) -> Component[VectorAssembler1Properties]:
        return newState

    class VectorAssembler1Code(ComponentCode):

        def __init__(self, props):
            self.props: VectorAssembler1.VectorAssembler1Properties = props

        def apply(self, spark: SparkSession, in0: DataFrame) -> DataFrame:
            from pyspark.ml.feature import VectorAssembler
            from pyspark.ml.regression import LinearRegression
            source_df = spark.read.options()
            vectorAssembler = VectorAssembler(inputCols=[x for x in self.props.selectColumns.split(",")], outputCol="features")
            vhouse_df = vectorAssembler.transform(in0)

            return vhouse_df
