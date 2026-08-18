import logging
from typing import List, Tuple

import libcst as cst

from semantic_transformer.core.transformation_name import TransformationName
from semantic_transformer.core.transformations.and_condition_splitter import (
    AndConditionSplitter,
)
from semantic_transformer.core.transformations.comparison_swapper import (
    ComparisonSwapper,
)
from semantic_transformer.core.transformations.dead_code_injector import (
    DeadCodeInjector,
)
from semantic_transformer.core.transformations.dead_method_injector import (
    DeadMethodTransformer,
)
from semantic_transformer.core.transformations.dead_string_assignment_injector import (
    StringFloodTransformer,
)
from semantic_transformer.core.transformations.double_negation_injector import (
    DoubleNegationInjector,
)
from semantic_transformer.core.transformations.for_loop_rewriting import (
    ForToWhileTransformer,
)
from semantic_transformer.core.transformations.if_else_switcher import IfElseSwitcher
from semantic_transformer.core.transformations.if_true_wrapper import IfTrueWrapper
from semantic_transformer.core.transformations.local_variable_renamer import (
    LocalVariableRenamer,
)
from semantic_transformer.core.transformations.operand_permuters import (
    CommutativeOperandPermuter,
)
from semantic_transformer.core.transformations.string_literal_splitter import (
    StringLiteralSplitter,
)
from semantic_transformer.core.transformations.try_except_injector import (
    TryExceptInjector,
)
from semantic_transformer.core.transformations.while_loop_unrolling import (
    WhileLoopUnroller,
)
from semantic_transformer.core.visitors.and_condition_visitor import AndConditionVisitor
from semantic_transformer.core.visitors.comparison_visitor import ComparisonVisitor
from semantic_transformer.core.visitors.dead_code_block_visitor import (
    DeadCodeBlockVisitor,
)
from semantic_transformer.core.visitors.dead_method_injection_visitor import (
    DeadMethodVisitor,
)
from semantic_transformer.core.visitors.dead_string_assignment_vistor import (
    StringFloodVisitor,
)
from semantic_transformer.core.visitors.double_negation_visitor import (
    DoubleNegationVisitor,
)
from semantic_transformer.core.visitors.for_loop_visitor import ForLoopVisitor
from semantic_transformer.core.visitors.if_else_visitor import IfElseVisitor
from semantic_transformer.core.visitors.indented_block_visitor import (
    IndentedBlockVisitor,
)
from semantic_transformer.core.visitors.local_variable_visitor import (
    LocalVariableVisitor,
)
from semantic_transformer.core.visitors.operand_permuter_visitor import (
    OperandPermuterVisitor,
)
from semantic_transformer.core.visitors.string_literal_visitor import (
    StringLiteralVisitor,
)
from semantic_transformer.core.visitors.while_visitor import WhileVisitor

TRANSFORMATION_MAP = {
    TransformationName.If_ELSE_SWAP.value: (IfElseVisitor, IfElseSwitcher),
    TransformationName.WHILE_LOOP_UNROLLING.value: (WhileVisitor, WhileLoopUnroller),
    TransformationName.FOR_LOOP_REWRITING.value: (
        ForLoopVisitor,
        ForToWhileTransformer,
    ),
    TransformationName.COMPARISON_SWAPPER.value: (ComparisonVisitor, ComparisonSwapper),
    TransformationName.LOCAL_VARIABLE_RENAMER.value: (LocalVariableVisitor, LocalVariableRenamer),
    TransformationName.COMMUTATIVE_OPERAND_PERMUTER.value: (
        OperandPermuterVisitor,
        CommutativeOperandPermuter,
    ),
    TransformationName.IF_TRUE_WRAPPER.value: (IndentedBlockVisitor, IfTrueWrapper),
    TransformationName.AND_CONDITION_SPLITTER.value: (AndConditionVisitor, AndConditionSplitter),
    TransformationName.DEAD_CODE_INJECTOR.value: (DeadCodeBlockVisitor, DeadCodeInjector),
    TransformationName.TRY_EXCEPT_INJECTOR.value: (
        IndentedBlockVisitor,
        TryExceptInjector,
    ),
    TransformationName.DOUBLE_NEGATION_INJECTOR.value: (
        DoubleNegationVisitor,
        DoubleNegationInjector,
    ),
    TransformationName.STRING_LITERAL_SPLITTER.value: (StringLiteralVisitor, StringLiteralSplitter),
    TransformationName.DEAD_STRING_ASSIGNMENT.value: (StringFloodVisitor, StringFloodTransformer),
    TransformationName.DEAD_METHHOD_INJECTION.value: (DeadMethodVisitor, DeadMethodTransformer),
}

ENCODING = "utf-8"
KEYWORD_SPECIFIC_TRANSFORMATIONS = [
    TransformationName.DEAD_STRING_ASSIGNMENT.value,
    TransformationName.STRING_LITERAL_SPLITTER.value,
    TransformationName.DEAD_METHHOD_INJECTION.value,
]


def collect_candidate_nodes(
    source_code: str, identifier: str, target_name: str = None
) -> List[Tuple[int, int]]:
    """
    Visits the source code and collects candidate nodes for transformation.

    Parameters
    ----------
    source_code : str
        The source code to be transformed.
    identifier : str
        The identifier for the type of transformation to apply.

    Returns
    -------
    List[Tuple[int, int]]
        The collected candidate nodes' coordinates.
    """
    if identifier not in TRANSFORMATION_MAP:
        raise ValueError(f"Unknown Identifier: {identifier}")

    visitor_cls, _ = TRANSFORMATION_MAP[identifier]
    if identifier == TransformationName.DEAD_METHHOD_INJECTION.value:
        visitor = visitor_cls(target_name=target_name)
    else:
        visitor = visitor_cls()

    try:
        wrapper = cst.MetadataWrapper(cst.parse_module(source_code))
        wrapper.visit(visitor)
        return visitor.candidates
    except Exception as e:
        logging.error(f"LibCST failed to parse source code: {e}")
        return []


def apply_transformation(
    source_code: str, target_coords: List, identifier: str, target_name: str | list[str] = None
) -> str:
    """
    Applies a transformation to the source code at the given target coordinates.

    Parameters
    ----------
    source_code : str
        The source code to be transformed.
    target_coords : List
        The list of target coordinates where the transformation should be applied.
    identifier : str
        The identifier for the type of transformation to apply.

    Returns
    -------
    str
        The transformed source code.
    """
    if identifier not in TRANSFORMATION_MAP:
        raise ValueError(f"Unknown Identifier: {identifier}")

    _, transformer_cls = TRANSFORMATION_MAP[identifier]
    if identifier in KEYWORD_SPECIFIC_TRANSFORMATIONS:
        transformer = transformer_cls(target_coords=set(target_coords), target_name=target_name)
    else:
        transformer = transformer_cls(target_coords=set(target_coords))

    try:
        wrapper = cst.MetadataWrapper(cst.parse_module(source_code))
        return wrapper.visit(transformer).code
    except Exception as e:
        logging.error(f"LibCST failed to apply transformation: {e}")
        return source_code
