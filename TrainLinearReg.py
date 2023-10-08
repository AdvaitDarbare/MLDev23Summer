from prophecy.cb.server.base.ComponentBuilderBase import *
from pyspark.sql import *
from pyspark.sql.functions import *
from prophecy.cb.server.base.datatypes import SInt
from prophecy.cb.ui.uispec import *
from prophecy.cb.ui.UISpecUtil import *


class TrainLinearReg(ComponentSpec):
    name: str = "TrainLinearReg"
    category: str = "Transform"

    def optimizeCode(self) -> bool:
        return True

    @dataclass(frozen=True)
    class TrainLinearRegProperties(ComponentProperties):
        targetColumn: Optional[str] = None
        maxIter: SInt = SInt("10")
        regParam: SInt = SInt("3")
        elasticNetParam: SInt = SInt("8")
        modelPath: str = ""

    def dialog(self) -> Dialog:
        return Dialog("Linear Regression").addElement(
            ColumnsLayout(gap="1rem", height="100%")
            .addColumn(PortSchemaTabs().importSchema(), "5fr")
            .addElement((SchemaColumnsDropdown("Target Column") \
                         .bindSchema("component.ports.inputs[0].schema")
                         .bindProperty("targetColumn")))
            .addElement((TextBox("Max Iterations", placeholder="10")
                         .bindProperty("maxIter")))
            .addElement((TextBox("RegParam", placeholder="3")
                         .bindProperty("regParam")))
            .addElement((TextBox("ElasticNetParam", placeholder="8")
                         .bindProperty("elasticNetParam")))
            .addElement((TextBox("Model Path", placeholder="Eg: dbfs:/FileStore/lr_model")
                         .bindProperty("modelPath")))
        )

    def validate(self, component: Component[TrainLinearRegProperties]) -> List[Diagnostic]:
        return []

    def onChange(self, oldState: Component[TrainLinearRegProperties], newState: Component[TrainLinearRegProperties]) -> Component[TrainLinearRegProperties]:
        return newState

    class TrainLinearRegCode(ComponentCode):

        def __init__(self, props):
            self.props: TrainLinearReg.TrainLinearRegProperties = props

        def apply(self, spark: SparkSession, in0: DataFrame):
            from pyspark.ml.regression import LinearRegression

            # Train the model
            lr = LinearRegression(featuresCol="features", labelCol=self.props.targetColumn, maxIter=self.props.maxIter,
                                  regParam=self.props.regParam, elasticNetParam=self.props.elasticNetParam / 100.0)

            lr_model = lr.fit(in0)
            # Save the model
            lr_model.write().overwrite().session(sparkSession=spark).save(self.props.modelPath)
