from enum import Enum


class TransformationName(Enum):
    If_ELSE_SWAP = "if_else_swap"
    WHILE_LOOP_UNROLLING = "while_loop_unrolling"
    COMMUTATIVE_OPERAND_PERMUTER = "commutative_operand_permuter"
    COMPARISON_SWAPPER = "comparison_swapper"
    FOR_LOOP_REWRITING = "for_loop_rewriting"
    FUNCTION_NAME_RENAMER = "function_name_renamer"
    IF_TRUE_WRAPPER = "if_true_wrapper"
    AND_CONDITION_SPLITTER = "and_condition_splitter"
    DEAD_CODE_INJECTOR = "dead_code_injector"
    TRY_EXCEPT_INJECTOR = "try_except_injector"
    DOUBLE_NEGATION_INJECTOR = "double_negation_injector"
    STRING_LITERAL_SPLITTER = "string_literal_splitter"
    DEAD_STRING_ASSIGNMENT = "dead_string_assignment"
    DEAD_METHHOD_INJECTION = "dead_method_injection"
    LOCAL_VARIABLE_RENAMER = "local_variable_renamer"
