import re

def analyze_complexity(code: str, language: str) -> str:
    """
    Analyzes the time and space complexity of the given code before optimization.
    Returns a string with the estimated complexities.
    """
    time_complexity = "O(n)"  # Default assumption
    space_complexity = "O(n)"  # Default assumption
    
    if language == "python":
        # Python-specific complexity analysis
        if "for i in range(len(" in code:
            time_complexity = "O(n)"
            space_complexity = "O(n)"
        if "enumerate(" in code:
            time_complexity = "O(n)"
            space_complexity = "O(1)"
    
    elif language == "js":
        # JavaScript-specific complexity analysis
        if "for (let i = 0;" in code:
            time_complexity = "O(n)"
            space_complexity = "O(n)"
        if "for (let i of" in code:
            time_complexity = "O(n)"
            space_complexity = "O(1)"
    
    elif language == "cpp":
        # C++-specific complexity analysis
        if "for (int i = 0;" in code:
            time_complexity = "O(n)"
            space_complexity = "O(n)"
        if "for (auto i :" in code:
            time_complexity = "O(n)"
            space_complexity = "O(1)"
    
    elif language == "java":
        # Java-specific complexity analysis
        if "for (int i = 0;" in code:
            time_complexity = "O(n)"
            space_complexity = "O(n)"
        if "for (int i : " in code:
            time_complexity = "O(n)"
            space_complexity = "O(1)"
    
    return time_complexity, space_complexity

def optimize_code(code: str) -> str:
    code = code.replace("\\n", "\n")

    # General loop replacement — even without *1 + 0
    pattern = r'for (\w+) in range\(len\((\w+)\)\):\n\s+(\w+).append\(\2\[\1\]\)'
    code = re.sub(pattern, r'for \1, item in enumerate(\2):\n    \3.append(item)', code)

    # Simplify math operations: x * 1 + 0 → x
    code = re.sub(r'\b(\w+) \* 1 \+ 0\b', r'\1', code)

    # Boolean simplifications
    code = re.sub(r'if (.+?) == False:', r'if not \1:', code)
    code = re.sub(r'if (.+?) == True:', r'if \1:', code)

    return code


def main():
    print("🔧 Welcome to Multi-Language Code Optimizer")
    language = input("Enter the language (python/js/cpp/java): ").lower()
    code_input = input(f"Paste your {language} code below (use \\n for new lines):\n")

    # Analyze time and space complexity before optimization
    before_time, before_space = analyze_complexity(code_input, language)
    
    # Optimize code
    optimized = optimize_code(code_input)
    
    # Analyze time and space complexity after optimization
    after_time, after_space = analyze_complexity(optimized, language)
    
    # Display results
    print("\n🚀 Optimized Code:\n")
    print(optimized)
    print("\n⏱ Time Complexity: {} → {}".format(before_time, after_time))
    print("🧠 Space Complexity: {} → {}".format(before_space, after_space))

if __name__ == "__main__":
    main()
