import ast
import re

# ================= 配置区 =================
UNARY_OPS = {
    'abs', 'inv', 'sqrt', 'log', 'log2', 'log10', 'exp',
    'sin', 'asin', 'sinh', 'asinh',
    'cos', 'acos', 'cosh', 'acosh',
    'tan', 'atan', 'tanh', 'atanh',
    'pow2', 'pow3', 'sign', 'neg'
}

BINARY_OPS = {
    'add', 'sub', 'mul', 'div', 'min', 'max', 'pow'
}

COMMUTATIVE_OPS = {'add', 'mul', 'max', 'min'}

MATH_CONSTANTS = {'e', 'pi', 'euler_gamma'}


class ExpressionConverter:
    def __init__(self):
        # [核心修改] 使用正则 Lookahead (?!\()
        # 意思：匹配单词 "add"，但仅当它后面"不是"左括号时才替换
        # 这样 add(a,b) 会保持原样，而 a add b 会变成 a + b
        self.str_replacements = [
            (r'\badd\s*(?!\()', '+'),
            (r'\bsub\s*(?!\()', '-'),
            (r'\bmul\s*(?!\()', '*'),
            (r'\bdiv\s*(?!\()', '/'),
            (r'\bmax\s*(?!\()', '|'),  # max -> BitOr
            (r'\bmin\s*(?!\()', '&'),  # min -> BitAnd
            (r'\bpow\s*(?!\()', '**'),  # pow -> Pow
        ]

        self.op_map = {
            ast.Add: 'add',
            ast.Sub: 'sub',
            ast.Mult: 'mul',
            ast.Div: 'div',
            ast.Pow: 'pow',
            ast.BitOr: 'max',  # 还原 |
            ast.BitAnd: 'min',  # 还原 &
        }

    def _preprocess_string(self, expr_str):
        # 预处理：应用智能替换
        for pattern, repl in self.str_replacements:
            expr_str = re.sub(pattern, repl, expr_str)
        return expr_str

    def _ast_to_prefix(self, node):
        # --- Case A: 二元运算 (BinOp: a + b 或 a | b) ---
        if isinstance(node, ast.BinOp):
            # [特殊处理] (x)**2 -> pow2(x), (x)**3 -> pow3(x)
            if isinstance(node.op, ast.Pow) and isinstance(node.right, (ast.Num, ast.Constant)):
                val = node.right.n if isinstance(node.right, ast.Num) else node.right.value
                if val == 2:
                    return ['pow2'] + self._ast_to_prefix(node.left)
                elif val == 3:
                    return ['pow3'] + self._ast_to_prefix(node.left)

            op_type = type(node.op)
            op_token = self.op_map.get(op_type, 'unknown_op')

            left_seq = self._ast_to_prefix(node.left)
            right_seq = self._ast_to_prefix(node.right)

            return self._finalize_binary_op(op_token, left_seq, right_seq)

        # --- Case B: 函数调用 (Call: add(a, b)) ---
        elif isinstance(node, ast.Call):
            func_name = node.func.id
            args_seq_list = [self._ast_to_prefix(arg) for arg in node.args]

            # 如果函数是二元且可交换 (如 max(a,b))，进行排序
            if func_name in COMMUTATIVE_OPS and len(args_seq_list) == 2:
                left_seq = args_seq_list[0]
                right_seq = args_seq_list[1]
                return self._finalize_binary_op(func_name, left_seq, right_seq)

            # 普通函数
            flat_args = []
            for seq in args_seq_list:
                flat_args.extend(seq)
            return [func_name] + flat_args

        # --- Case C: 常数 ---
        elif isinstance(node, (ast.Constant, ast.Num)):
            return ['<C>']

        # --- Case D: 变量 & 数学常数 ---
        elif isinstance(node, ast.Name):
            if node.id in MATH_CONSTANTS:
                # [修改] 保留原名，不要返回 ['<C>']
                return [node.id]
            return [node.id]

        # --- Case E: 一元运算 ---
        elif isinstance(node, ast.UnaryOp):
            if isinstance(node.op, ast.USub):
                if isinstance(node.operand, (ast.Num, ast.Constant)):
                    return ['<C>']  # -5 -> <C>
                return ['neg'] + self._ast_to_prefix(node.operand)  # -x -> neg x
            return ['unknown_unary'] + self._ast_to_prefix(node.operand)

        else:
            raise ValueError(f"Unsupported AST node type: {type(node)}")

    def _finalize_binary_op(self, op_token, left_seq, right_seq):
        """辅助排序"""
        if op_token in COMMUTATIVE_OPS:
            left_str = " ".join(left_seq)
            right_str = " ".join(right_seq)
            if left_str > right_str:
                return [op_token] + right_seq + left_seq
        return [op_token] + left_seq + right_seq

    def convert(self, raw_expression):
        clean_expr = self._preprocess_string(raw_expression)
        try:
            tree = ast.parse(clean_expr, mode='eval')
        except SyntaxError as e:
            # 打印出替换后的字符串，方便调试
            raise ValueError(f"解析失败: {e} | 原始输入: {raw_expression} | 替换后: {clean_expr}")
        return self._ast_to_prefix(tree.body)


class StructureCanonicalizer:
    def __init__(self):
        pass

    def get_canonical_skeleton(self, expression_list):
        tree, _ = self._parse(expression_list, 0)
        return self._canonicalize_node(tree).split(' ')

    def _parse(self, tokens, idx):
        if idx >= len(tokens):
            raise ValueError(f"Unexpected end of expression. Tokens: {tokens}")

        token = tokens[idx]
        idx += 1

        if token.startswith('x_'):
            return {'type': 'var', 'val': token}, idx

        # B. [修改] 数学常数 (视为特殊叶子节点)
        if token in MATH_CONSTANTS:
            # 这里的 type 可以叫 'math_const' 也可以混入 'op' (arity=0)，看你喜好
            # 建议给个独立 type 以便后续处理
            return {'type': 'math_const', 'val': token}, idx

        if token == '<C>' or self._is_number(token) or token in MATH_CONSTANTS:
            return {'type': 'const', 'val': '<C>'}, idx

        if token in UNARY_OPS:
            arity = 1
        elif token in BINARY_OPS:
            arity = 2
        else:
            arity = 2

        children = []
        for _ in range(arity):
            child, idx = self._parse(tokens, idx)
            children.append(child)

        return {'type': 'op', 'val': token, 'children': children}, idx

    def _canonicalize_node(self, node):
        if node['type'] == 'const':
            return '<C>'
        if node['type'] == 'var':
            return node['val']
        # [新增] 数学常数直接返回名字
        if node['type'] == 'math_const':
            return node['val']

        children_strs = [self._canonicalize_node(child) for child in node['children']]

        if node['val'] in COMMUTATIVE_OPS:
            children_strs.sort()

        return f"{node['val']} " + " ".join(children_strs)

    def _is_number(self, s):
        try:
            float(s)
            return True
        except ValueError:
            return False


# ================= 测试 =================
if __name__ == "__main__":
    converter = ExpressionConverter()
    canon = StructureCanonicalizer()

    print("=== 数学常数 & 规范化测试 ===\n")

    test_cases = [
        # Case 1: 基础常数识别 (pi)
        # 预期: pi 被转为 <C>，且 <C> (ASCII 60) 排在 x_0 (ASCII 120) 前面
        # Raw: add(x_0, pi) -> Token: [add, x_0, <C>] -> Sort: add <C> x_0
        "add(x_0, pi)",

        # Case 2: 欧拉常数 (e) 与乘法
        # 预期: e -> <C>, add(x_1, x_0) -> add x_0 x_1
        # mul(e, ...) -> mul <C> add x_0 x_1
        "mul(e, add(x_1, x_0))",

        # Case 3: 欧拉-马歇罗尼常数 (euler_gamma)
        # 预期: euler_gamma -> <C>
        # sub 不是交换律算子，顺序保持不变: sub <C> x_2
        "sub(euler_gamma, x_2)",

        # Case 4: 嵌套结构排序
        # 内部: add(pi, x_1) -> add <C> x_1
        # 外部: add(x_0, 内部)
        # 比较: "x_0" vs "add <C> x_1"
        # 字符串序: "a"(add) < "x"(x_0)，所以内部结构排在前面
        # 结果: add add <C> x_1 x_0
        "add(x_0, add(pi, x_1))",

        # Case 5: 混合数字和数学常数
        # 3.14 -> <C>, pi -> <C>
        # 结果: add <C> <C> (顺序取决于 converter 解析顺序，通常由原始顺序决定，除非完全一样)
        # 注意: 如果两个子节点规范化后字符串完全一样(都是 <C>)，排序不改变相对位置
        "add(3.14, pi)",

        # Case 6: 复杂函数中的常数
        # pow(pi, x_0) -> pow <C> x_0 (pow 不可交换)
        "pow(pi, x_0)"
    ]

    for i, raw in enumerate(test_cases, 1):
        print(f"Case {i}: {raw}")
        try:
            tokens = converter.convert(raw)
            skel = canon.get_canonical_skeleton(tokens)
            print(f"  Tokens: {tokens}")
            print(f"  Canon : {skel}")
        except Exception as e:
            print(f"  Error : {e}")
        print("-" * 30)