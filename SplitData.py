from prophecy.cb.server.base.ComponentBuilderBase import *
from pyspark.sql import *
from pyspark.sql.functions import *
from prophecy.cb.server.base.datatypes import SInt
from prophecy.cb.ui.uispec import *
from prophecy.cb.ui.UISpecUtil import *


class SplitData(ComponentSpec):
    name: str = "SplitData"
    category: str = "Transform"

    def optimizeCode(self) -> bool:
        return True

    @dataclass(frozen=True)
    class SplitDataProperties(ComponentProperties):
        trainSplit: Optional[SFloat] = None
        testSplit: Optional[SFloat] = None


    def dialog(self) -> Dialog:
        return Dialog("Linear Regression").addElement(
            ColumnsLayout(gap="1rem", height="100%")
            .addColumn(PortSchemaTabs().importSchema(), "2fr")
            .addElement((TextBox("Train Split units (%)", placeholder="70")
                         .bindProperty("trainSplit")))
            .addElement((TextBox("Test Split units (%)", placeholder="30")
                         .bindProperty("testSplit")))
        )

    def validate(self, component: Component[SplitDataProperties]) -> List[Diagnostic]:
        return []

    def onChange(self, oldState: Component[SplitDataProperties], newState: Component[SplitDataProperties]) -> Component[SplitDataProperties]:
        return newState

    class SplitDataCode(ComponentCode):

        def __init__(self, props):
            self.props: SplitData.SplitDataProperties = props

        def apply(self, spark: SparkSession, in0: DataFrame) -> (DataFrame, DataFrame):
            train_df, test_df = in0.randomSplit([self.props.trainSplit, self.props.testSplit])

            return (train_df, test_df)
