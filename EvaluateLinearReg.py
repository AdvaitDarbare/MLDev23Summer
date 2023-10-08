from prophecy.cb.server.base.ComponentBuilderBase import *
from pyspark.sql import *
from pyspark.sql.functions import *

from prophecy.cb.server.base.datatypes import *
from prophecy.cb.ui.uispec import *
from prophecy.cb.ui.UISpecUtil import *

from pyspark.sql.functions import array, struct, lit, col



class EvaluateLinearReg(ComponentSpec):
    name: str = "EvaluateLinearReg"
    category: str = "Transform"

    def optimizeCode(self) -> bool:
        return True

    @dataclass(frozen=True)
    class EvaluateLinearRegProperties(ComponentProperties):
        targetColumn: Optional[str] = None

    def dialog(self) -> Dialog:
        return Dialog("Linear Regression").addElement(
            ColumnsLayout(gap="1rem", height="100%")
            .addColumn(PortSchemaTabs().importSchema(), "2fr")
            .addElement((SchemaColumnsDropdown("Target Column") \
                         .bindSchema("component.ports.inputs[0].schema")
                         .bindProperty("targetColumn")))
        )

    def validate(self, component: Component[EvaluateLinearRegProperties]) -> List[Diagnostic]:
        return list()


    def onChange(self, oldState: Component[EvaluateLinearRegProperties], newState: Component[EvaluateLinearRegProperties]) -> Component[
        EvaluateLinearRegProperties]:
        return newState


    class EvaluateLinearRegCode(ComponentCode):
        def __init__(self, newProps, in0: DataFrame):
            self.props: EvaluateLinearReg.EvaluateLinearRegProperties = newProps

        def apply(self, spark: SparkSession, in0: DataFrame):
            from pyspark.ml.evaluation import RegressionEvaluator

            evaluator = RegressionEvaluator(labelCol=self.props.targetColumn, predictionCol="prediction", metricName="rmse")
            rmse = evaluator.evaluate(in0)

            print("Root Mean Squared Error (RMSE) on test data:", rmse)
