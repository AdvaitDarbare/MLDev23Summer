from prophecy.cb.server.base.ComponentBuilderBase import *
from pyspark.sql import *
from pyspark.sql.functions import *

from prophecy.cb.server.base.datatypes import *
from prophecy.cb.ui.uispec import *
from prophecy.cb.ui.UISpecUtil import *

from pyspark.sql.functions import array, struct, lit, col



class PredictLinearReg(ComponentSpec):
    name: str = "PredictLinearReg"
    category: str = "Transform"

    def optimizeCode(self) -> bool:
        return True

    @dataclass(frozen=True)
    class PredictLinearRegProperties(ComponentProperties):
        modelPath: str = ""
        targetColumn: str = ""

    def dialog(self) -> Dialog:
        return Dialog("Linear Regression").addElement(
            ColumnsLayout(gap="1rem", height="100%")
            .addColumn(PortSchemaTabs().importSchema(), "2fr").addElement((TextBox("Model Path", placeholder="Eg: dbfs:/FileStore/lr_model")
                                                                           .bindProperty("modelPath")))
            .addElement((TextBox("Target Column", placeholder=" ")
                         .bindProperty("targetColumn")))

        )

    def validate(self, component: Component[PredictLinearRegProperties]) -> List[Diagnostic]:
        return list()


    def onChange(self, oldState: Component[PredictLinearRegProperties], newState: Component[PredictLinearRegProperties]) -> Component[
        PredictLinearRegProperties]:
        return newState


    class PredictLinearRegCode(ComponentCode):
        def __init__(self, newProps, in0: DataFrame):
            self.props: PredictLinearReg.PredictLinearRegProperties = newProps

        def apply(self, spark: SparkSession, in0: DataFrame) -> DataFrame:
            from pyspark.ml.regression import LinearRegressionModel

            out0 = LinearRegressionModel.load(self.props.modelPath).transform(in0)

            return out0
