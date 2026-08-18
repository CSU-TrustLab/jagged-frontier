from semantic_transformer.api import apply_transformation, collect_candidate_nodes
from semantic_transformer.core.transformation_name import TransformationName

source_code = """
from file_A import result

module_var = 1

class MyClass:
    class_var = 2
    
    def my_method(self, x):
        \"\"\"This is a docstring.\"\"\"
        a = 10
        if a > 5:
            b = 20
            return b
            c = 30
        else:
            d = 40
        
        def inner_func():
            y = 2
            try:
                z = 3
                raise ValueError()
                w = 4
            except ValueError:
                pass
    
    def injected_plot_func(darray, **kwargs):
        row = "x" 

        allargs = locals().copy() 
        allargs.update(allargs.pop("kwargs"))  

        allargs.pop("darray", None)
        allargs.pop("row", None)

        return mock_matplotlib_scatter(kind="plot1d", **allargs)
"""

print("=== String Flood ===")
spt = TransformationName.DEAD_STRING_ASSIGNMENT.value
candidates = collect_candidate_nodes(source_code, spt)
print(candidates)
print(apply_transformation(source_code, candidates, spt, "content_length"))
