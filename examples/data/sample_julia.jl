# Sample Julia code with optimization opportunities

# Inefficient fibonacci implementation
function inefficient_fibonacci(n::Int)
    if n <= 1
        return n
    end
    return inefficient_fibonacci(n - 1) + inefficient_fibonacci(n - 2)
end

# Struct with hardcoded values and type instability issues
mutable struct DataProcessor
    batch_size::Int      # Hardcoded value
    timeout::Float64     # Hardcoded value
    counter::Int
    
    function DataProcessor()
        new(100, 5.0, 0)  # Hardcoded configuration
    end
end

# Multiple passes through data (inefficient)
function process_data(processor::DataProcessor, data::Vector{Dict{String, Any}})
    result = Dict{String, Any}[]
    
    # First pass - filter active items
    for item in data
        if haskey(item, "active") && item["active"] == true
            push!(result, item)
        end
    end
    
    # Second pass - calculate totals
    total = 0
    for item in data
        if haskey(item, "value")
            total += item["value"]  # Type instability - could be Any
        end
    end
    
    # Third pass - transform items
    transformed = Dict{String, Any}[]
    for item in result
        new_item = transform_item(processor, item)
        push!(transformed, new_item)
    end
    
    # Magic number usage
    if length(transformed) > 50
        println("Warning: Large dataset")
    end
    
    return transformed
end

# No error handling or type stability
function transform_item(processor::DataProcessor, item::Dict{String, Any})
    transformed = Dict{String, Any}()
    
    # No validation - could throw KeyError
    transformed["id"] = item["id"]
    transformed["name"] = uppercase(item["name"])
    transformed["timestamp"] = time()
    
    # Increment counter unsafely
    processor.counter += 1
    
    return transformed
end

# Inefficient string building
function build_report(items::Vector{Dict{String, Any}})
    report = ""
    
    # Inefficient string concatenation in loop
    for (i, item) in enumerate(items)
        report *= "Item $i: $(get(item, "name", "Unknown"))\n"
        report *= "Status: $(get(item, "status", "N/A"))\n"
        report *= "---\n"
    end
    
    return report
end

# Global variable (not recommended)
global_counter = 0

function increment_global_counter()
    global global_counter
    global_counter += 1
    return global_counter
end

# Function with poor error handling
function process_file(filename::String)
    # No error handling for file operations
    content = read(filename, String)
    return content
end

# Type unstable function
function calculate_something(x)  # No type annotation
    if x > 10
        return x * 2.5  # Returns Float64
    else
        return x        # Returns input type (could be Int)
    end
end