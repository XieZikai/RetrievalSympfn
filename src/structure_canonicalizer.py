import ast
import re

import ast
import re

import ast
import re


class ExpressionConverter:
    def __init__(self):
        # 1. 字符串替换策略 (The Hack)
        # 我们利用 Python 的位运算符来“借尸还魂”
        # max -> | (BitOr)
        # min -> & (BitAnd)
        # 这样 (a max b) 变成 (a | b)，Python AST 就能解析了！
        self.str_replacements = [
            (r'\badd\b', '+'),
            (r'\bsub\b', '-'),
            (r'\bmul\b', '*'),
            (r'\bdiv\b', '/'),
            (r'\bmax\b', '|'),  # <--- 关键修改
            (r'\bmin\b', '&'),  # <--- 关键修改
        ]

        # 2. AST 节点映射
        self.op_map = {
            ast.Add: 'add',
            ast.Sub: 'sub',
            ast.Mult: 'mul',
            ast.Div: 'div',
            ast.Pow: 'pow',
            ast.BitOr: 'max',  # <--- 还原 max
            ast.BitAnd: 'min',  # <--- 还原 min
        }

        # 3. 交换律集合 (max 和 min 也满足交换律)
        self.commutative_ops = {'add', 'mul', 'max', 'min'}

    def _preprocess_string(self, expr_str):
        for pattern, repl in self.str_replacements:
            expr_str = re.sub(pattern, repl, expr_str)
        return expr_str

    def _ast_to_prefix(self, node):
        # --- Case A: 二元运算 (BinOp) ---
        if isinstance(node, ast.BinOp):
            op_type = type(node.op)
            op_token = self.op_map.get(op_type, 'unknown_op')

            left_seq = self._ast_to_prefix(node.left)
            right_seq = self._ast_to_prefix(node.right)

            # 结构规范化：交换律算子排序
            if op_token in self.commutative_ops:
                left_str = " ".join(left_seq)
                right_str = " ".join(right_seq)
                if left_str > right_str:
                    return [op_token] + right_seq + left_seq

            return [op_token] + left_seq + right_seq

        # --- Case C: 常数 ---
        elif isinstance(node, (ast.Constant, ast.Num)):
            return ['<C>']

        # --- Case D: 变量 ---
        elif isinstance(node, ast.Name):
            return [node.id]

        # --- Case E: 一元运算 ---
        elif isinstance(node, ast.UnaryOp):
            if isinstance(node.op, ast.USub):
                return self._ast_to_prefix(node.operand)
            # 处理 inv (倒数) -> 假设映射为 Python 的 ~ (Invert) 或者其他
            # 如果你的数据里有 'inv(...)' 这种写法，它会被解析为 ast.Call 而不是 UnaryOp
            # 如果是 'neg ...' 这种，看情况
            return ['unknown_unary'] + self._ast_to_prefix(node.operand)

        # --- Case F: 函数调用 (你的数据里有 sqrt, inv 等) ---
        elif isinstance(node, ast.Call):
            func_name = node.func.id
            args_seq = []
            for arg in node.args:
                args_seq.extend(self._ast_to_prefix(arg))
            # 函数参数通常不排序，除非你也想让 sqrt(x) 和 sqrt(y) 排序
            # 但 inv, sqrt 都是一元函数，不需要排序
            return [func_name] + args_seq

        else:
            raise ValueError(f"Unsupported AST node type: {type(node)}")

    def convert(self, raw_expression):
        clean_expr = self._preprocess_string(raw_expression)
        try:
            tree = ast.parse(clean_expr, mode='eval')
        except SyntaxError as e:
            print(f"预处理后的字符串: {clean_expr}")
            raise ValueError(f"解析失败: {e}")

        prefix_seq = self._ast_to_prefix(tree.body)
        return prefix_seq


class StructureCanonicalizer:
    def __init__(self):
        # 满足交换律的算子
        self.commutative_ops = {'add', 'mul'}

    def get_canonical_skeleton(self, expression_list):
        """
        入口: 输入前缀列表，例如 ['add', 'x_1', 'x_0']
        输出: 规范化字符串 "add x_0 x_1"
        """
        # 1. 解析成树结构 (递归)
        tree, _ = self._parse(expression_list, 0)

        # 2. 规范化并转字符串
        return self._canonicalize_node(tree)

    def _parse(self, tokens, idx):
        """简单的递归解析器"""
        if idx >= len(tokens):
            raise ValueError("Unexpected end of expression. Structrue might be invalid.")

        token = tokens[idx]
        idx += 1

        # A. 变量 (x_0 ... x_15)
        if token.startswith('x_'):
            return {'type': 'var', 'val': token}, idx

        # B. [修改点] 常数
        # 这里必须同时支持 原始数字字符串 和 已经被替换过的 '<C>'
        if token == '<C>' or self._is_number(token):
            return {'type': 'const', 'val': token}, idx

        # C. 算子
        # 假设都是二元算子，除了 sin/cos/exp/log/sqrt 是的一元算子
        # 注意：pow 是二元算子 (**)，会走 else 分支，这也是正确的
        arity = 1 if token in ['sin', 'cos', 'exp', 'log', 'sqrt', 'tan', 'abs', 'max', 'min'] else 2

        children = []
        for _ in range(arity):
            child, idx = self._parse(tokens, idx)
            children.append(child)

        return {'type': 'op', 'val': token, 'children': children}, idx

    def _canonicalize_node(self, node):
        # 1. 忽略常数：所有数字变成统一的 <C>
        if node['type'] == 'const':
            return '<C>'

        # 2. 变量：直接返回 x_i
        if node['type'] == 'var':
            return node['val']

        # 3. 递归处理子节点
        children_strs = [self._canonicalize_node(child) for child in node['children']]

        # 4. === 核心逻辑：排序 ===
        # 如果是加法或乘法，必须对子节点的字符串表示进行排序
        # 这自动涵盖了你说的 "x_0 在 x_1 前"，也涵盖了复杂子树的排序
        if node['val'] in self.commutative_ops:
            children_strs.sort()

            # 5. 拼接返回
        return f"{node['val']} " + " ".join(children_strs)

    def _is_number(self, s):
        try:
            float(s)
            return True
        except ValueError:
            return False


if __name__ == "__main__":
    # --- 测试 ---
    canon = StructureCanonicalizer()

    # Case 1: 简单变量交换 (你提到的情况)
    # x_1 + x_0 -> x_0 + x_1
    expr1 = ['add', 'x_1', 'x_0']
    print(f"Case 1: {canon.get_canonical_skeleton(expr1)}")
    # 输出: add x_0 x_1 (正确)

    # Case 2: 复杂嵌套 (必须比较子树字符串的情况)
    # (x_0 * x_2) + (x_0 * x_1)
    # 原始前缀: add mul x_0 x_2 mul x_0 x_1
    expr2 = ['add', 'mul', 'x_0', 'x_2', 'mul', 'x_0', 'x_1']
    print(f"Case 2: {canon.get_canonical_skeleton(expr2)}")
    # 输出: add mul x_0 x_1 mul x_0 x_2
    # (正确！它发现了 mul x_0 x_1 应该排在 mul x_0 x_2 前面)

    converter = ExpressionConverter()

    # 你的原始数据
    raw_data = "(-0.352 add (cos((-2.07 add (-1.32 mul x_0))) mul ((88.0 add (2.04 mul x_1)))**3))"

    print(f"原始输入: {raw_data}\n")

    result = converter.convert(raw_data)
    print(f"转换结果 (Token List): {result}")
    print(f"转换结果 (String): {' '.join(result)}")

    # 验证是否包含了 <C>
    print(f"\n包含 <C> 数量: {result.count('<C>')}")

    print(f"\n转换为canonical：{canon.get_canonical_skeleton(result)}")
