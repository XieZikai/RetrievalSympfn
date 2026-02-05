from abc import ABC, abstractmethod
# from ast import parse
# from operator import length_hint, xor
import numpy as np
import scipy.special
import copy
from copy import deepcopy
# from numpy.compat.py3k import npy_load_module
from collections import defaultdict
from scipy.stats import special_ortho_group
import argparse
from argparse import Namespace
from structure_canonicalizer import StructureCanonicalizer, ExpressionConverter
from tokenizer import FormulaTokenizer


################################################原来的utils.py###########################################################
def chunks(lst, n):
    """Yield successive n-sized chunks from lst."""
    for i in range(0, len(lst), n):
        yield lst[i: i + n]


##############################################原来的encoder##############################################################
class Encoder(ABC):
    """
    Base class for encoders, encodes and decodes matrices
    abstract methods for encoding/decoding numbers
    """

    def __init__(self, params):
        pass

    @abstractmethod
    def encode(self, val):
        pass

    @abstractmethod
    def decode(self, lst):
        pass


class GeneralEncoder:
    def __init__(self, params, symbols, all_operators):
        self.float_encoder = FloatSequences(params)
        self.equation_encoder = Equation(
            params, symbols, self.float_encoder, all_operators
        )


# 定义 FloatSequences 类，继承自 Encoder 类
class FloatSequences(Encoder):
    def __init__(self, params):
        super().__init__(params)
        self.float_precision = params.float_precision
        self.mantissa_len = params.mantissa_len  # 尾数长度
        self.max_exponent = params.max_exponent  # 最大指数
        self.base = (self.float_precision + 1) // self.mantissa_len  # 基数
        self.max_token = 10 ** self.base  # 最大令牌数
        self.symbols = ["+", "-"]
        self.symbols.extend(
            ["N" + f"%0{self.base}d" % i for i in range(self.max_token)]
        )  # 添加 N0, N1, ..., N9999 等符号
        self.symbols.extend(
            ["E" + str(i) for i in range(-self.max_exponent, self.max_exponent + 1)]
        )  # 添加 E-5, E-4, ..., E5 等符号

    # 实现 encode 方法，用于编码浮点数
    def encode(self, values):
        """
        Write a float number
        """
        # 获取浮点数精度
        precision = self.float_precision
        # 如果输入值是一维数组
        if len(values.shape) == 1:
            seq = []
            value = values
            for val in value:  # 遍历数组中的每个值
                assert val not in [-np.inf, np.inf]  # 确保值不是无穷大
                sign = "+" if val >= 0 else "-"
                m, e = (f"%.{precision}e" % val).split("e")  # 将值转换为科学计数法字符串，并分割为尾数和指数
                i, f = m.lstrip("-").split(".")  # 分割尾数为整数部分和小数部分
                i = i + f
                tokens = chunks(i, self.base)  # 将尾数按基数分割为令牌
                expon = int(e) - precision  # 计算指数
                if expon < -self.max_exponent:  # 如果指数小于最小指数，将令牌设为零，并将指数设为零
                    tokens = ["0" * self.base] * self.mantissa_len
                    expon = int(0)
                seq.extend([sign, *["N" + token for token in tokens], "E" + str(expon)])  # 将符号、令牌和指数添加到序列中
            return seq
        else:  # 如果输入值是多维数组，递归调用 encode 方法
            seqs = [self.encode(values[0])]
            N = values.shape[0]
            for n in range(1, N):
                seqs += [self.encode(values[n])]
        return seqs

    def decode(self, lst):  # 实现 decode 方法，用于解码浮点数序列
        """
        Parse a list that starts with a float.
        Return the float value, and the position it ends in the list.
        """
        if len(lst) == 0:
            return None
        seq = []
        for val in chunks(lst, 2 + self.mantissa_len):  # 按固定长度分割列表
            for x in val:  # 检查每个元素的首字符是否为合法字符

                if x[0] not in ["-", "+", "E", "N"]:
                    return np.nan
            try:
                sign = 1 if val[0] == "+" else -1
                mant = ""
                for x in val[1:-1]:
                    mant += x[1:]
                # 将尾数转换为整数
                mant = int(mant)
                # 获取指数
                exp = int(val[-1][1:])
                # 计算浮点数的值
                value = sign * mant * (10 ** exp)
                value = float(value)
            except Exception:
                # 如果出现异常，返回 NaN
                value = np.nan
            seq.append(value)
        return seq


class Equation(Encoder):
    def __init__(self, params, symbols, float_encoder, all_operators):
        super().__init__(params)
        self.params = params  # 保存参数
        self.max_int = self.params.max_int  # 最大整数
        self.symbols = symbols  # 符号列表
        if params.extra_unary_operators != "":
            self.extra_unary_operators = self.params.extra_unary_operators.split(",")
        else:
            self.extra_unary_operators = []
        if params.extra_binary_operators != "":
            self.extra_binary_operators = self.params.extra_binary_operators.split(",")
        else:
            self.extra_binary_operators = []
        self.float_encoder = float_encoder
        self.all_operators = all_operators

    # 实现 encode 方法，用于编码方程树
    def encode(self, tree):
        res = []
        for elem in tree.prefix().split(","):  # 遍历方程树的前缀表示法
            try:  # 将元素转换为浮点数
                val = float(elem)
                if elem.lstrip("-").isdigit():  # 如果是整数，调用 write_int 方法编码
                    res.extend(self.write_int(int(elem)))
                else:  # 如果是浮点数，调用 float_encoder 的 encode 方法编码
                    res.extend(self.float_encoder.encode(np.array([val])))
            except ValueError:  # 如果不是数字，直接添加到结果列表中
                res.append(elem)
        return res

    def _decode(self, lst):
        if len(lst) == 0:
            return None, 0
        # elif (lst[0] not in self.symbols) and (not lst[0].lstrip("-").replace(".","").replace("e+", "").replace("e-","").isdigit()):
        #     return None, 0
        elif "OOD" in lst[0]:  # 如果列表的第一个元素包含 "OOD"，返回 None 和 0
            return None, 0
        elif lst[0] in self.all_operators.keys():  # 如果列表的第一个元素是运算符
            res = Node(lst[0], self.params)  # 创建一个节点
            arity = self.all_operators[lst[0]]  # 获取运算符的元数
            pos = 1
            for i in range(arity):  # 递归解码子节点
                child, length = self._decode(lst[pos:])
                if child is None:
                    return None, pos
                res.push_child(child)
                pos += length
            return res, pos
        elif lst[0].startswith("INT"):  # 如果列表的第一个元素以 "INT" 开头
            val, length = self.parse_int(lst)  # 解析整数
            return Node(str(val), self.params), length
        elif lst[0] == "+" or lst[0] == "-":  # 如果列表的第一个元素是正负号
            try:  # 解码浮点数
                val = self.float_encoder.decode(lst[:3])[0]
            except Exception as e:
                # print(e, "error in encoding, lst: {}".format(lst))
                return None, 0
            return Node(str(val), self.params), 3
        elif (
                lst[0].startswith("CONSTANT") or lst[0] == "y"  # 如果列表的第一个元素以 "CONSTANT" 开头或为 "y"
        ):  ##added this manually CAREFUL!!
            return Node(lst[0], self.params), 1
        elif lst[0] in self.symbols:
            return Node(lst[0], self.params), 1
        else:
            try:
                float(lst[0])  # if number, return leaf
                return Node(lst[0], self.params), 1
            except:
                return None, 0

    def split_at_value(self, lst, value):  # 按指定值分割列表
        indices = [i for i, x in enumerate(lst) if x == value]  # 获取指定值的索引
        res = []
        for start, end in zip(  # 按索引分割列表
                [0, *[i + 1 for i in indices]], [*[i - 1 for i in indices], len(lst)]
        ):
            res.append(lst[start: end + 1])
        return res

    def decode(self, lst):  # 实现 decode 方法，用于解码方程序列
        trees = []
        lists = self.split_at_value(lst, "|")  # 按 "|" 分割列表
        for lst in lists:  # 解码每个子列表
            tree = self._decode(lst)[0]
            if tree is None:
                return None
            trees.append(tree)
        tree = NodeList(trees)  # 创建节点列表
        return tree

    def parse_int(self, lst):  # 解析整数
        """
        Parse a list that starts with an integer.
        Return the integer value, and the position it ends in the list.
        """
        base = self.max_int
        val = 0
        i = 0
        for x in lst[1:]:  # 遍历列表中的元素
            if not (x.rstrip("-").isdigit()):
                break
            val = val * base + int(x)  # 计算整数的值
            i += 1
        if base > 0 and lst[0] == "INT-":  # 如果是负数，取反
            val = -val
        return val, i + 1

    def write_int(self, val):  # 将整数转换为指定基数的表示
        """
        Convert a decimal integer to a representation in the given base.
        """
        if not self.params.use_sympy:  # 如果不使用 sympy
            return [str(val)]

        base = self.max_int  # 基数
        res = []
        max_digit = abs(base)  # 最大数字
        neg = val < 0  # 判断是否为负数
        val = -val if neg else val
        while True:
            rem = val % base
            val = val // base
            if rem < 0 or rem > max_digit:
                rem -= base
                val += 1
            res.append(str(rem))
            if val == 0:
                break
        res.append("INT-" if neg else "INT+")  # 添加符号
        return res[::-1]


###########################################################原来的generator.py#############################################


SPECIAL_WORDS = [
    "<EOS>",
    "<X>",
    "</X>",
    "<Y>",
    "</Y>",
    "</POINTS>",
    "<INPUT_PAD>",
    "<OUTPUT_PAD>",
    "<PAD>",
    "(",
    ")",
    "SPECIAL",
    "OOD_unary_op",
    "OOD_binary_op",
    "OOD_constant",
]
operators_real = {
    "add": 2,
    "sub": 2,
    "mul": 2,
    "div": 2,
    "min": 2,
    "max": 2,
    "abs": 1,
    "inv": 1,
    "sqrt": 1,
    "log": 1,
    "log2": 1,  # new
    "log10": 1,  # new
    "exp": 1,
    "sin": 1,
    "asin": 1,  # revise
    "sinh": 1,  # new
    "asinh": 1,  # new
    "cos": 1,
    "acos": 1,  # revise
    "cosh": 1,  # new
    "acosh": 1,  # new
    "tan": 1,
    "atan": 1,  # revise
    "tanh": 1,  # new
    "atanh": 1,  # new
    "pow2": 1,
    "pow3": 1,
    "sign": 1,  # new
}
operators_extra = {"pow": 2}
math_constants = ["e", "pi", "euler_gamma", "CONSTANT"]
all_operators = {**operators_real, **operators_extra}


class Node:
    def __init__(self, value, params, children=None):
        self.value = value
        self.children = children if children else []
        self.params = params

    def push_child(self, child):
        self.children.append(child)

    def prefix(self):
        s = str(self.value)
        for c in self.children:
            s += "," + c.prefix()
        return s

    # export to latex qtree format: prefix with \Tree, use package qtree
    def qtree_prefix(self):
        s = "[.$" + str(self.value) + "$ "
        for c in self.children:
            s += c.qtree_prefix()
        s += "]"
        return s

    def infix(self):
        nb_children = len(self.children)
        if nb_children == 0:
            if self.value.lstrip("-").isdigit():
                return str(self.value)
            else:
                # try:
                #    s = f"%.{self.params.float_precision}e" % float(self.value)
                # except ValueError:
                s = str(self.value)
                return s
        if nb_children == 1:
            s = str(self.value)
            if s == "pow2":
                s = "(" + self.children[0].infix() + ")**2"
            elif s == "pow3":
                s = "(" + self.children[0].infix() + ")**3"
            else:
                s = s + "(" + self.children[0].infix() + ")"
            return s
        s = "(" + self.children[0].infix()
        for c in self.children[1:]:
            s = s + " " + str(self.value) + " " + c.infix()
        return s + ")"

    def __len__(self):
        lenc = 1
        for c in self.children:
            lenc += len(c)
        return lenc

    def __str__(self):
        # infix a default print
        return self.infix()

    def __repr__(self):
        # infix a default print
        return str(self)

    def val(self, x, deterministic=True):
        if len(self.children) == 0:
            if str(self.value).startswith("x_"):
                _, dim = self.value.split("_")
                dim = int(dim)
                return x[:, dim]
            elif str(self.value) == "rand":
                if deterministic:
                    return np.zeros((x.shape[0],))
                return np.random.randn(x.shape[0])
            elif str(self.value) in math_constants:
                return getattr(np, str(self.value)) * np.ones((x.shape[0],))
            else:
                return float(self.value) * np.ones((x.shape[0],))

        if self.value == "add":
            return self.children[0].val(x) + self.children[1].val(x)
        if self.value == "sub":
            return self.children[0].val(x) - self.children[1].val(x)
        if self.value == "mul":
            m1, m2 = self.children[0].val(x), self.children[1].val(x)
            try:
                return m1 * m2
            except Exception as e:
                # print(e)
                nans = np.empty((m1.shape[0],))
                nans[:] = np.nan
                return nans
        if self.value == "pow":
            m1, m2 = self.children[0].val(x), self.children[1].val(x)
            try:
                return np.power(m1, m2)
            except Exception as e:
                # print(e)
                nans = np.empty((m1.shape[0],))
                nans[:] = np.nan
                return nans
        if self.value == "max":
            return np.maximum(self.children[0].val(x), self.children[1].val(x))
        if self.value == "min":
            return np.minimum(self.children[0].val(x), self.children[1].val(x))

        if self.value == "div":
            denominator = self.children[1].val(x)
            denominator[denominator == 0.0] = np.nan
            try:
                return self.children[0].val(x) / denominator
            except Exception as e:
                # print(e)
                nans = np.empty((denominator.shape[0],))
                nans[:] = np.nan
                return nans

        if self.value == "inv":
            denominator = self.children[0].val(x)
            denominator[denominator == 0.0] = np.nan
            try:
                return 1 / denominator
            except Exception as e:
                # print(e)
                nans = np.empty((denominator.shape[0],))
                nans[:] = np.nan
                return nans
        if self.value == "log":
            numerator = self.children[0].val(x)
            if self.params.use_abs:
                numerator[numerator <= 0.0] *= -1
            else:
                numerator[numerator <= 0.0] = np.nan
            try:
                return np.log(numerator)
            except Exception as e:
                # print(e)
                nans = np.empty((numerator.shape[0],))
                nans[:] = np.nan
                return nans

        if self.value == "log2":
            numerator = self.children[0].val(x)
            if self.params.use_abs:
                numerator[numerator <= 0.0] *= -1
            else:
                numerator[numerator <= 0.0] = np.nan
            try:
                return np.log2(numerator)
            except Exception as e:
                # print(e)
                nans = np.empty((numerator.shape[0],))
                nans[:] = np.nan
                return nans

        if self.value == "log10":
            numerator = self.children[0].val(x)
            if self.params.use_abs:
                numerator[numerator <= 0.0] *= -1
            else:
                numerator[numerator <= 0.0] = np.nan
            try:
                return np.log10(numerator)
            except Exception as e:
                # print(e)
                nans = np.empty((numerator.shape[0],))
                nans[:] = np.nan
                return nans
        if self.value == "exp":  ## new
            return np.exp(self.children[0].val(x))

        if self.value == "sin":  ## new
            return np.sin(self.children[0].val(x))

        if self.value == "asin":  ## new
            return np.arcsin(self.children[0].val(x))

        if self.value == "sinh":  ## new
            return np.sinh(self.children[0].val(x))

        if self.value == "asinh":  ## new
            return np.arcsinh(self.children[0].val(x))

        if self.value == "cos":  ## new
            return np.cos(self.children[0].val(x))

        if self.value == "acos":  ## new
            return np.arccos(self.children[0].val(x))

        if self.value == "cosh":  ## new
            return np.cosh(self.children[0].val(x))

        if self.value == "acosh":  ## new
            return np.arccosh(self.children[0].val(x))

        if self.value == "tan":  ## new
            return np.tan(self.children[0].val(x))

        if self.value == "atan":  ## new
            return np.arctan(self.children[0].val(x))

        if self.value == "tanh":  ## new
            return np.tanh(self.children[0].val(x))

        if self.value == "atanh":  ## new
            return np.arctanh(self.children[0].val(x))

        if self.value == "sqrt":
            numerator = self.children[0].val(x)
            if self.params.use_abs:
                numerator[numerator <= 0.0] *= -1
            else:
                numerator[numerator < 0.0] = np.nan
            try:
                return np.sqrt(numerator)
            except Exception as e:
                # print(e)
                nans = np.empty((numerator.shape[0],))
                nans[:] = np.nan
                return nans
        if self.value == "pow2":
            numerator = self.children[0].val(x)
            try:
                return numerator ** 2
            except Exception as e:
                # print(e)
                nans = np.empty((numerator.shape[0],))
                nans[:] = np.nan
                return nans
        if self.value == "pow3":
            numerator = self.children[0].val(x)
            try:
                return numerator ** 3
            except Exception as e:
                # print(e)
                nans = np.empty((numerator.shape[0],))
                nans[:] = np.nan
                return nans
        if self.value == "abs":
            return np.abs(self.children[0].val(x))
        if self.value == "sign":
            return (self.children[0].val(x) >= 0) * 2.0 - 1.0
        if self.value == "step":
            x = self.children[0].val(x)
            return x if x > 0 else 0
        if self.value == "id":
            return self.children[0].val(x)
        if self.value == "fresnel":
            return scipy.special.fresnel(self.children[0].val(x))[0]
        if self.value.startswith("eval"):
            n = self.value[-1]
            return getattr(scipy.special, self.value[:-1])(n, self.children[0].val(x))[
                0
            ]
        else:
            fn = getattr(np, self.value, None)
            if fn is not None:
                try:
                    return fn(self.children[0].val(x))
                except Exception as e:
                    nans = np.empty((x.shape[0],))
                    nans[:] = np.nan
                    return nans
            fn = getattr(scipy.special, self.value, None)
            if fn is not None:
                return fn(self.children[0].val(x))
            assert False, "Could not find function"

    def get_recurrence_degree(self):
        recurrence_degree = 0
        if len(self.children) == 0:
            if str(self.value).startswith("x_"):
                _, _, offset = self.value.split("_")
                offset = int(offset)
                if offset > recurrence_degree:
                    recurrence_degree = offset
            return recurrence_degree
        return max([child.get_recurrence_degree() for child in self.children])

    def replace_node_value(self, old_value, new_value):
        if self.value == old_value:
            self.value = new_value
        for child in self.children:
            child.replace_node_value(old_value, new_value)


class NodeList:
    def __init__(self, nodes):
        self.nodes = []
        for node in nodes:
            self.nodes.append(node)
        self.params = nodes[0].params

    def infix(self):
        return " | ".join([node.infix() for node in self.nodes])

    def __len__(self):
        return sum([len(node) for node in self.nodes])

    def prefix(self):
        return ",|,".join([node.prefix() for node in self.nodes])

    def __str__(self):
        return self.infix()

    def __repr__(self):
        return str(self)

    def val(self, xs, deterministic=True):
        batch_vals = [
            np.expand_dims(node.val(np.copy(xs), deterministic=deterministic), -1)
            for node in self.nodes
        ]
        return np.concatenate(batch_vals, -1)

    def replace_node_value(self, old_value, new_value):
        for node in self.nodes:
            node.replace_node_value(old_value, new_value)


class Generator(ABC):
    def __init__(self, params):
        pass

    @abstractmethod
    def generate_datapoints(self, rng):
        pass


class RandomFunctions(Generator):
    def __init__(self, params, special_words):
        super().__init__(params)
        self.params = params
        self.prob_const = params.prob_const
        self.prob_rand = params.prob_rand
        self.max_int = params.max_int
        self.min_binary_ops_per_dim = params.min_binary_ops_per_dim
        self.max_binary_ops_per_dim = params.max_binary_ops_per_dim
        self.min_unary_ops = params.min_unary_ops
        self.max_unary_ops = params.max_unary_ops
        self.min_output_dimension = params.min_output_dimension
        self.min_input_dimension = params.min_input_dimension
        self.max_input_dimension = params.max_input_dimension
        self.max_output_dimension = params.max_output_dimension
        self.max_number = 10 ** (params.max_exponent + params.float_precision)
        self.operators = copy.deepcopy(operators_real)

        self.operators_dowsample_ratio = defaultdict(float)
        if params.operators_to_downsample != "":
            for operator in self.params.operators_to_downsample.split(","):
                operator, ratio = operator.split("_")
                ratio = float(ratio)
                self.operators_dowsample_ratio[operator] = ratio

        if params.required_operators != "":
            self.required_operators = self.params.required_operators.split(",")
        else:
            self.required_operators = []

        if params.extra_binary_operators != "":
            self.extra_binary_operators = self.params.extra_binary_operators.split(",")
        else:
            self.extra_binary_operators = []
        if params.extra_unary_operators != "":
            self.extra_unary_operators = self.params.extra_unary_operators.split(",")
        else:
            self.extra_unary_operators = []

        self.unaries = [
                           o for o in self.operators.keys() if np.abs(self.operators[o]) == 1
                       ] + self.extra_unary_operators

        self.binaries = [
                            o for o in self.operators.keys() if np.abs(self.operators[o]) == 2
                        ] + self.extra_binary_operators

        unaries_probabilities = []
        for op in self.unaries:
            if op not in self.operators_dowsample_ratio:
                unaries_probabilities.append(1.0)
            else:
                ratio = self.operators_dowsample_ratio[op]
                unaries_probabilities.append(ratio)
        self.unaries_probabilities = np.array(unaries_probabilities)
        self.unaries_probabilities /= self.unaries_probabilities.sum()

        binaries_probabilities = []
        for op in self.binaries:
            if op not in self.operators_dowsample_ratio:
                binaries_probabilities.append(1.0)
            else:
                ratio = self.operators_dowsample_ratio[op]
                binaries_probabilities.append(ratio)
        self.binaries_probabilities = np.array(binaries_probabilities)
        self.binaries_probabilities /= self.binaries_probabilities.sum()

        self.unary = False  # len(self.unaries) > 0
        self.distrib = self.generate_dist(
            2 * self.max_binary_ops_per_dim * self.max_input_dimension
        )

        self.constants = [
            str(i) for i in range(-self.max_int, self.max_int + 1) if i != 0
        ]
        self.constants += math_constants
        self.variables = ["rand"] + [f"x_{i}" for i in range(self.max_input_dimension)]
        self.symbols = (
                list(self.operators)
                + self.constants
                + self.variables
                + ["|", "INT+", "INT-", "FLOAT+", "FLOAT-", "pow", "0"]
        )
        self.constants.remove("CONSTANT")

        if self.params.extra_constants is not None:
            self.extra_constants = self.params.extra_constants.split(",")
        else:
            self.extra_constants = []

        self.general_encoder = GeneralEncoder(
            params, self.symbols, all_operators
        )
        self.float_encoder = self.general_encoder.float_encoder
        self.float_words = special_words + sorted(list(set(self.float_encoder.symbols)))
        self.equation_encoder = self.general_encoder.equation_encoder
        self.equation_words = sorted(list(set(self.symbols)))
        self.equation_words = special_words + self.equation_words

    def generate_dist(self, max_ops):
        """
        `max_ops`: maximum number of operators
        Enumerate the number of possible unary-binary trees that can be generated from empty nodes.
        D[e][n] represents the number of different binary trees with n nodes that
        can be generated from e empty nodes, using the following recursion:
            D(n, 0) = 0
            D(0, e) = 1
            D(n, e) = D(n, e - 1) + p_1 * D(n- 1, e) + D(n - 1, e + 1)
        p1 =  if binary trees, 1 if unary binary
        """
        p1 = 1 if self.unary else 0
        # enumerate possible trees
        D = []
        D.append([0] + ([1 for i in range(1, 2 * max_ops + 1)]))
        for n in range(1, 2 * max_ops + 1):  # number of operators
            s = [0]
            for e in range(1, 2 * max_ops - n + 1):  # number of empty nodes
                s.append(s[e - 1] + p1 * D[n - 1][e] + D[n - 1][e + 1])
            D.append(s)
        assert all(
            len(D[i]) >= len(D[i + 1]) for i in range(len(D) - 1)
        ), "issue in generate_dist"
        return D

    def generate_float(self, rng, exponent=None):
        sign = rng.choice([-1, 1])
        mantissa = float(rng.choice(range(1, 10 ** self.params.float_precision)))
        min_power = (
                -self.params.max_exponent_prefactor - (self.params.float_precision + 1) // 2
        )
        max_power = (
                self.params.max_exponent_prefactor - (self.params.float_precision + 1) // 2
        )
        if not exponent:
            exponent = rng.randint(min_power, max_power + 1)
        constant = sign * (mantissa * 10 ** exponent)
        return str(constant)

    def generate_int(self, rng):
        return str(rng.choice(self.constants + self.extra_constants))

    def generate_leaf(self, rng, input_dimension):
        if rng.rand() < self.prob_rand:
            return "rand"
        else:
            if self.n_used_dims < input_dimension:
                dimension = self.n_used_dims
                self.n_used_dims += 1
                return f"x_{dimension}"
            else:
                draw = rng.rand()
                if draw < self.prob_const:
                    return self.generate_int(rng)
                else:
                    dimension = rng.randint(0, input_dimension)
                    return f"x_{dimension}"

    def generate_ops(self, rng, arity):
        if arity == 1:
            ops = self.unaries
            probas = self.unaries_probabilities
        else:
            ops = self.binaries
            probas = self.binaries_probabilities
        return rng.choice(ops, p=probas)

    def sample_next_pos(self, rng, nb_empty, nb_ops):
        """
        Sample the position of the next node (binary case).
        Sample a position in {0, ..., `nb_empty` - 1}.
        """
        assert nb_empty > 0
        assert nb_ops > 0
        probs = []
        if self.unary:
            for i in range(nb_empty):
                probs.append(self.distrib[nb_ops - 1][nb_empty - i])
        for i in range(nb_empty):
            probs.append(self.distrib[nb_ops - 1][nb_empty - i + 1])
        probs = [p / self.distrib[nb_ops][nb_empty] for p in probs]
        probs = np.array(probs, dtype=np.float64)
        e = rng.choice(len(probs), p=probs)
        arity = 1 if self.unary and e < nb_empty else 2
        e %= nb_empty
        return e, arity

    def generate_tree(self, rng, nb_binary_ops, input_dimension):
        self.n_used_dims = 0
        tree = Node(0, self.params)
        empty_nodes = [tree]
        next_en = 0
        nb_empty = 1
        while nb_binary_ops > 0:
            next_pos, arity = self.sample_next_pos(rng, nb_empty, nb_binary_ops)
            next_en += next_pos
            op = self.generate_ops(rng, arity)
            empty_nodes[next_en].value = op
            for _ in range(arity):
                e = Node(0, self.params)
                empty_nodes[next_en].push_child(e)
                empty_nodes.append(e)
            next_en += 1
            nb_empty += arity - 1 - next_pos
            nb_binary_ops -= 1
        rng.shuffle(empty_nodes)
        for n in empty_nodes:
            if len(n.children) == 0:
                n.value = self.generate_leaf(rng, input_dimension)
        return tree

    ##########################################################检查min(x,x)或max(x,x)##########################################################
    def _check_no_trivial_minmax(self, node):
        """
        返回 False 表示发现了 min(x, x) 或 max(x, x)
        """
        if node.value in ["min", "max"]:
            if len(node.children) == 2:
                if node.children[0].prefix() == node.children[1].prefix():
                    return False
        for c in node.children:
            if not self._check_no_trivial_minmax(c):
                return False
        return True

    def generate_multi_dimensional_tree(
            self,
            rng,
            input_dimension=None,
            output_dimension=None,
            nb_unary_ops=None,
            nb_binary_ops=None,
    ):
        trees = []

        if input_dimension is None:
            input_dimension = rng.randint(
                self.min_input_dimension, self.max_input_dimension + 1
            )
        if output_dimension is None:
            output_dimension = rng.randint(
                self.min_output_dimension, self.max_output_dimension + 1
            )
        if nb_binary_ops is None:
            min_binary_ops = self.min_binary_ops_per_dim * input_dimension
            max_binary_ops = self.max_binary_ops_per_dim * input_dimension
            nb_binary_ops_to_use = [
                rng.randint(
                    min_binary_ops, self.params.max_binary_ops_offset + max_binary_ops
                )
                for dim in range(output_dimension)
            ]
        elif isinstance(nb_binary_ops, int):
            nb_binary_ops_to_use = [nb_binary_ops for _ in range(output_dimension)]
        else:
            nb_binary_ops_to_use = nb_binary_ops
        if nb_unary_ops is None:
            nb_unary_ops_to_use = [
                rng.randint(self.min_unary_ops, self.max_unary_ops + 1)
                for dim in range(output_dimension)
            ]
        elif isinstance(nb_unary_ops, int):
            nb_unary_ops_to_use = [nb_unary_ops for _ in range(output_dimension)]
        else:
            nb_unary_ops_to_use = nb_unary_ops

        for i in range(output_dimension):
            tree = self.generate_tree(rng, nb_binary_ops_to_use[i], input_dimension)
            tree = self.add_unaries(rng, tree, nb_unary_ops_to_use[i])
            ##Adding constants
            if self.params.reduce_num_constants:
                tree = self.add_prefactors(rng, tree)
            else:
                tree = self.add_linear_transformations(rng, tree, target=self.variables)
                tree = self.add_linear_transformations(rng, tree, target=self.unaries)
            trees.append(tree)
        tree = NodeList(trees)

        nb_unary_ops_to_use = [
            len([x for x in tree_i.prefix().split(",") if x in self.unaries])
            for tree_i in tree.nodes
        ]
        nb_binary_ops_to_use = [
            len([x for x in tree_i.prefix().split(",") if x in self.binaries])
            for tree_i in tree.nodes
        ]

        for op in self.required_operators:
            if op not in tree.infix():
                return self.generate_multi_dimensional_tree(
                    rng, input_dimension, output_dimension, nb_unary_ops, nb_binary_ops
                )

        ####################################################################### check: forbid min(x, x) / max(x, x)##############
        for tree_i in tree.nodes:
            if not self._check_no_trivial_minmax(tree_i):
                return self.generate_multi_dimensional_tree(
                    rng, input_dimension, output_dimension, nb_unary_ops, nb_binary_ops
                )

        return (
            tree,
            input_dimension,
            output_dimension,
            nb_unary_ops_to_use,
            nb_binary_ops_to_use,
        )

    def add_unaries(self, rng, tree, nb_unaries):
        prefix = self._add_unaries(rng, tree)
        prefix = prefix.split(",")
        indices = []
        for i, x in enumerate(prefix):
            if x in self.unaries:
                indices.append(i)
        rng.shuffle(indices)
        if len(indices) > nb_unaries:
            to_remove = indices[: len(indices) - nb_unaries]
            for index in sorted(to_remove, reverse=True):
                del prefix[index]
        tree = self.equation_encoder.decode(prefix).nodes[0]
        return tree

    def _add_unaries(self, rng, tree):

        s = str(tree.value)

        for c in tree.children:
            if len(c.prefix().split(",")) < self.params.max_unary_depth:
                unary = rng.choice(self.unaries, p=self.unaries_probabilities)
                s += f",{unary}," + self._add_unaries(rng, c)
            else:
                s += f"," + self._add_unaries(rng, c)
        return s

    def add_prefactors(self, rng, tree):
        transformed_prefix = self._add_prefactors(rng, tree)
        if transformed_prefix == tree.prefix():
            a = self.generate_float(rng)
            transformed_prefix = f"mul,{a}," + transformed_prefix
        a = self.generate_float(rng)
        transformed_prefix = f"add,{a}," + transformed_prefix
        tree = self.equation_encoder.decode(transformed_prefix.split(",")).nodes[0]
        return tree

    def _add_prefactors(self, rng, tree):

        s = str(tree.value)
        a, b = self.generate_float(rng), self.generate_float(rng)
        if s in ["add", "sub"]:
            s += (
                     "," if tree.children[0].value in ["add", "sub"] else f",mul,{a},"
                 ) + self._add_prefactors(rng, tree.children[0])
            s += (
                     "," if tree.children[1].value in ["add", "sub"] else f",mul,{b},"
                 ) + self._add_prefactors(rng, tree.children[1])
        elif s in self.unaries and tree.children[0].value not in ["add", "sub"]:
            s += f",add,{a},mul,{b}," + self._add_prefactors(rng, tree.children[0])
        else:
            for c in tree.children:
                s += f"," + self._add_prefactors(rng, c)
        return s

    def add_linear_transformations(self, rng, tree, target, add_after=False):

        prefix = tree.prefix().split(",")
        indices = []

        for i, x in enumerate(prefix):
            if x in target:
                indices.append(i)

        offset = 0
        for idx in indices:
            a, b = self.generate_float(rng), self.generate_float(rng)
            if add_after:
                prefix = (
                        prefix[: idx + offset + 1]
                        + ["add", a, "mul", b]
                        + prefix[idx + offset + 1:]
                )
            else:
                prefix = (
                        prefix[: idx + offset]
                        + ["add", a, "mul", b]
                        + prefix[idx + offset:]
                )
            offset += 4
        tree = self.equation_encoder.decode(prefix).nodes[0]

        return tree

    def relabel_variables(self, tree):
        active_variables = []
        for elem in tree.prefix().split(","):
            if elem.startswith("x_"):
                active_variables.append(elem)
        active_variables = list(set(active_variables))
        input_dimension = len(active_variables)
        if input_dimension == 0:
            return 0
        active_variables.sort(key=lambda x: int(x[2:]))
        for j, xi in enumerate(active_variables):
            tree.replace_node_value(xi, "x_{}".format(j))
        return input_dimension

    def function_to_skeleton(
            self, tree, skeletonize_integers=False, constants_with_idx=False
    ):
        constants = []
        prefix = tree.prefix().split(",")
        j = 0
        for i, pre in enumerate(prefix):
            try:
                float(pre)
                is_float = True
                if pre.lstrip("-").isdigit():
                    is_float = False
            except ValueError:
                is_float = False

            if pre.startswith("CONSTANT"):
                constants.append("CONSTANT")
                if constants_with_idx:
                    prefix[i] = "CONSTANT_{}".format(j)
                j += 1
            elif is_float or (pre is self.constants and skeletonize_integers):
                if constants_with_idx:
                    prefix[i] = "CONSTANT_{}".format(j)
                else:
                    prefix[i] = "CONSTANT"
                while i > 0 and prefix[i - 1] in self.unaries:
                    del prefix[i - 1]
                try:
                    value = float(pre)
                except:
                    value = getattr(np, pre)
                constants.append(value)
                j += 1
            else:
                continue

        new_tree = self.equation_encoder.decode(prefix)
        return new_tree, constants

    def wrap_equation_floats(self, tree, constants):
        tree = self.tree
        env = self.env
        prefix = tree.prefix().split(",")
        j = 0
        for i, elem in enumerate(prefix):
            if elem.startswith("CONSTANT"):
                prefix[i] = str(constants[j])
                j += 1
        assert j == len(constants), "all constants were not fitted"
        assert "CONSTANT" not in prefix, "tree {} got constant after wrapper {}".format(
            tree, constants
        )
        tree_with_constants = env.word_to_infix(prefix, is_float=False, str_array=False)
        return tree_with_constants

    def order_datapoints(self, inputs, outputs):
        mean_input = inputs.mean(0)
        distance_to_mean = np.linalg.norm(inputs - mean_input, axis=-1)
        order_by_distance = np.argsort(distance_to_mean)
        return inputs[order_by_distance], outputs[order_by_distance]

    def _generate_datapoints(
            self,
            tree,
            n_points,
            scale,
            rng,
            input_dimension,
            input_distribution_type,
            n_centroids,
            max_trials,
            rotate=True,
            offset=None,
    ):
        inputs, outputs = [], []
        # 我们需要收集够 n_points 个点
        remaining_points = n_points
        trials = 0

        # 预先生成分布参数，避免在 while 循环里重复生成，保持分布一致性
        means = rng.randn(n_centroids, input_dimension, )
        covariances = rng.uniform(0, 1, size=(n_centroids, input_dimension))
        if rotate:
            rotations = [special_ortho_group.rvs(input_dimension) if input_dimension > 1 else np.identity(1) for i in
                         range(n_centroids)]
        else:
            rotations = [np.identity(input_dimension) for i in range(n_centroids)]
        weights = rng.uniform(0, 1, size=(n_centroids,))
        weights /= np.sum(weights)

        while remaining_points > 0 and trials < max_trials:
            # 动态调整本次生成的数量：
            # 为了保险起见，每次稍微多生成一点 (1.2倍)，增加命中率
            n_samples_needed = int(remaining_points * 1.5) + 10

            # 重新采样权重分布
            n_points_comp = rng.multinomial(n_samples_needed, weights)

            # === 生成 Input X ===
            if input_distribution_type == "gaussian":
                input_batch = np.vstack([
                    rng.multivariate_normal(mean, np.diag(covariance), int(sample)) @ rotation
                    for (mean, covariance, rotation, sample) in zip(means, covariances, rotations, n_points_comp)
                ])
            elif input_distribution_type == "uniform":
                input_batch = np.vstack([
                    (mean + rng.uniform(-1, 1, size=(sample, input_dimension)) * np.sqrt(covariance)) @ rotation
                    for (mean, covariance, rotation, sample) in zip(means, covariances, rotations, n_points_comp)
                ])

            # 预处理 Input
            input_std = np.std(input_batch, axis=0, keepdims=True)
            input_std[input_std < 1e-9] = 1.0
            input_batch = (input_batch - np.mean(input_batch, axis=0, keepdims=True)) / input_std
            input_batch *= scale
            if offset is not None:
                mean_off, std_off = offset
                input_batch *= std_off
                input_batch += mean_off

            # === 计算 Output Y ===
            # 注意：这里可能会产生警告，可以忽略，我们靠后面的 mask 过滤
            with np.errstate(all='ignore'):
                output_batch = tree.val(input_batch)

            # === [核心] 强力过滤逻辑 ===
            if output_batch.ndim == 1:
                output_batch = output_batch.reshape(-1, 1)

            # 1. 处理复数: 只取实部，虚部大的标记为 NaN
            if np.iscomplexobj(output_batch):
                mask_complex = np.abs(output_batch.imag) > 1e-6
                output_batch = output_batch.real
                output_batch[mask_complex] = np.nan

            # 2. 统一转 float64 处理 NaN/Inf
            output_batch = output_batch.astype(np.float64)

            # 3. 生成 Mask (有限值 & 范围内)
            mask_finite = np.all(np.isfinite(output_batch), axis=1)
            mask_range = np.all(np.abs(output_batch) < self.max_number, axis=1)
            # 同时也检查 input 是否正常
            mask_input = np.all(np.isfinite(input_batch), axis=1)

            final_mask = mask_finite & mask_range & mask_input

            # === 应用过滤 ===
            valid_x = input_batch[final_mask]
            valid_y = output_batch[final_mask]

            if valid_y.shape[0] > 0:
                inputs.append(valid_x)
                outputs.append(valid_y)
                remaining_points -= valid_y.shape[0]

            trials += 1

        # 循环结束
        if remaining_points > 0:
            # 尝试了 max_trials 次依然没凑够点，说明这个公式很难搞 (bad domain)
            # 返回 None 让上层决定重采公式
            return None, None

        # 拼接并截取正好 n_points 个
        final_inputs = np.concatenate(inputs, 0)[:n_points]
        final_outputs = np.concatenate(outputs, 0)[:n_points]

        # 展平 Y
        if final_outputs.shape[1] == 1:
            final_outputs = final_outputs.flatten()

        return final_inputs, final_outputs

    def generate_datapoints(
            self,
            tree,
            n_input_points,
            n_prediction_points,
            prediction_sigmas,
            rotate=True,
            offset=None,
            **kwargs,
    ):
        inputs, outputs = self._generate_datapoints(
            tree=tree,
            n_points=n_input_points,
            scale=1,
            rotate=rotate,
            offset=offset,
            **kwargs,
        )

        if inputs is None:
            return None, None
        datapoints = {"fit": (inputs, outputs)}

        if n_prediction_points == 0:
            return tree, datapoints
        for sigma_factor in prediction_sigmas:
            inputs, outputs = self._generate_datapoints(
                tree=tree,
                n_points=n_prediction_points,
                scale=sigma_factor,
                rotate=rotate,
                offset=offset,
                **kwargs,
            )
            if inputs is None:
                return None, None
            datapoints["predict_{}".format(sigma_factor)] = (inputs, outputs)

        return tree, datapoints


###################################################微调运算符##############################################################
def perturb_tree_operators(generator, tree, n, rng):
    """
    只替换 n 个 operator（保持 arity 不变）
    """
    prefix = tree.prefix().split(",")

    # 找出 operator 的位置
    op_indices = []
    for i, tok in enumerate(prefix):
        if tok in generator.unaries or tok in generator.binaries:
            op_indices.append(i)

    if len(op_indices) == 0:
        return tree

    n = min(n, len(op_indices))
    chosen = rng.choice(op_indices, size=n, replace=False)

    new_prefix = prefix.copy()
    for idx in chosen:
        tok = prefix[idx]
        if tok in generator.unaries:
            candidates = [o for o in generator.unaries if o != tok]
        else:
            candidates = [o for o in generator.binaries if o != tok]
        if len(candidates) > 0:
            new_prefix[idx] = rng.choice(candidates)

    new_tree = generator.equation_encoder.decode(new_prefix).nodes[0]
    return new_tree


######################################################utils.py#############################################################
FALSY_STRINGS = {"off", "false", "0"}
TRUTHY_STRINGS = {"on", "true", "1"}


def bool_flag(s):
    """
    Parse boolean arguments from the command line.
    """
    if s.lower() in FALSY_STRINGS:
        return False
    elif s.lower() in TRUTHY_STRINGS:
        return True
    else:
        raise argparse.ArgumentTypeError("Invalid value for a boolean flag!")


############################################################params#########################################################
def get_parser():
    """
    Generate a parameters parser.
    """

    parser = argparse.ArgumentParser(description="Function prediction", add_help=False)

    parser.add_argument(
        "--queue_strategy",
        type=str,
        default="uniform_sampling",
        help="in [precompute_batches, uniform_sampling, uniform_sampling_replacement]",
    )

    parser.add_argument("--collate_queue_size", type=int, default=2000)

    parser.add_argument(
        "--use_sympy",
        type=bool_flag,
        default=False,
        help="Whether to use sympy parsing (basic simplification)",
    )
    parser.add_argument(
        "--simplify",
        type=bool_flag,
        default=False,
        help="Whether to use further sympy simplification",
    )
    parser.add_argument(
        "--use_abs",
        type=bool_flag,
        default=False,
        help="Whether to replace log and sqrt by log(abs) and sqrt(abs)",
    )

    # encoding
    parser.add_argument(
        "--operators_to_downsample",
        type=str,
        default="div_0,arcsin_0,arccos_0,tan_0.2,arctan_0.2,sqrt_5,pow2_3,inv_3",
        help="Which operator to remove",
    )
    parser.add_argument(
        "--operators_to_not_repeat",
        type=str,
        default="",
        help="Which operator to not repeat",
    )

    parser.add_argument(
        "--max_unary_depth",
        type=int,
        default=6,
        help="Max number of operators inside unary",
    )

    parser.add_argument(
        "--required_operators",
        type=str,
        default="",
        help="Which operator to remove",
    )
    parser.add_argument(
        "--extra_unary_operators",
        type=str,
        default="",
        help="Extra unary operator to add to data generation",
    )
    parser.add_argument(
        "--extra_binary_operators",
        type=str,
        default="",
        help="Extra binary operator to add to data generation",
    )
    parser.add_argument(
        "--extra_constants",
        type=str,
        default=None,
        help="Additional int constants floats instead of ints",
    )

    parser.add_argument("--min_input_dimension", type=int, default=1)
    parser.add_argument("--max_input_dimension", type=int, default=16)
    parser.add_argument("--min_output_dimension", type=int, default=1)
    parser.add_argument("--max_output_dimension", type=int, default=1)
    parser.add_argument(
        "--enforce_dim",
        type=bool,
        default=True,
        help="should we enforce that we get as many examples of each dim ?",
    )

    parser.add_argument(
        "--use_controller",
        type=bool,
        default=True,
        help="should we enforce that we get as many examples of each dim ?",
    )

    parser.add_argument(
        "--float_precision",
        type=int,
        default=3,
        help="Number of digits in the mantissa",
    )
    parser.add_argument(
        "--mantissa_len",
        type=int,
        default=1,
        help="Number of tokens for the mantissa (must be a divisor or float_precision+1)",
    )
    parser.add_argument(
        "--max_exponent", type=int, default=100, help="Maximal order of magnitude"
    )
    parser.add_argument(
        "--max_exponent_prefactor",
        type=int,
        default=1,
        help="Maximal order of magnitude in prefactors",
    )
    parser.add_argument(
        "--max_token_len",
        type=int,
        default=0,
        help="max size of tokenized sentences, 0 is no filtering",
    )
    parser.add_argument(
        "--tokens_per_batch",
        type=int,
        default=10000,
        help="max number of tokens per batch",
    )
    parser.add_argument(
        "--pad_to_max_dim",
        type=bool,
        default=True,
        help="should we pad inputs to the maximum dimension?",
    )

    # generator
    parser.add_argument(
        "--max_int",
        type=int,
        default=10,
        help="Maximal integer in symbolic expressions",
    )
    parser.add_argument(
        "--min_binary_ops_per_dim",
        type=int,
        default=0,
        help="Min number of binary operators per input dimension",
    )
    parser.add_argument(
        "--max_binary_ops_per_dim",
        type=int,
        default=1,
        help="Max number of binary operators per input dimension",
    )
    parser.add_argument(
        "--max_binary_ops_offset",
        type=int,
        default=4,
        help="Offset for max number of binary operators",
    )
    parser.add_argument(
        "--min_unary_ops", type=int, default=0, help="Min number of unary operators"
    )
    parser.add_argument(
        "--max_unary_ops",
        type=int,
        default=4,
        help="Max number of unary operators",
    )
    parser.add_argument(
        "--min_op_prob",
        type=float,
        default=0.01,
        help="Minimum probability of generating an example with given n_op, for our curriculum strategy",
    )
    parser.add_argument(
        "--max_len", type=int, default=200, help="Max number of terms in the series"
    )
    parser.add_argument(
        "--min_len_per_dim", type=int, default=5, help="Min number of terms per dim"
    )
    parser.add_argument(
        "--max_centroids",
        type=int,
        default=10,
        help="Max number of centroids for the input distribution",
    )
    parser.add_argument(
        "--prob_const",
        type=float,
        default=0.0,
        help="Probability to generate integer in leafs",
    )
    parser.add_argument(
        "--reduce_num_constants",
        type=bool,
        default=True,
        help="Use minimal amount of constants in eqs",
    )
    parser.add_argument(
        "--use_skeleton",
        type=bool,
        default=False,
        help="should we use a skeleton rather than functions with constants",
    )
    parser.add_argument(
        "--prob_rand",
        type=float,
        default=0.0,
        help="Probability to generate n in leafs",
    )
    parser.add_argument(
        "--max_trials",
        type=int,
        default=1,
        help="How many trials we have for a given function",
    )
    parser.add_argument(
        "--n_prediction_points",
        type=int,
        default=200,
        help="number of next terms to predict",
    )
    parser.add_argument(
        "--prediction_sigmas",
        type=str,
        default="1,2,4,8,16",
        help="sigmas value for generation predicts",
    )
    return parser


DEFAULTS = dict(
    queue_strategy="uniform_sampling",
    collate_queue_size=2000,
    use_sympy=False,
    simplify=False,
    use_abs=False,
    operators_to_downsample="div_0,arcsin_0,arccos_0,tan_0.2,arctan_0.2,sqrt_5,pow2_3,inv_3",
    operators_to_not_repeat="",
    max_unary_depth=6,
    required_operators="",
    extra_unary_operators="",
    extra_binary_operators="",
    extra_constants=None,
    min_input_dimension=1,
    max_input_dimension=16,
    min_output_dimension=1,
    max_output_dimension=1,
    enforce_dim=True,
    use_controller=True,
    float_precision=3,
    mantissa_len=1,
    max_exponent=100,
    max_exponent_prefactor=1,
    max_token_len=0,
    tokens_per_batch=10000,
    pad_to_max_dim=True,
    max_int=10,
    min_binary_ops_per_dim=0,
    max_binary_ops_per_dim=1,
    max_binary_ops_offset=4,
    min_unary_ops=0,
    max_unary_ops=4,
    min_op_prob=0.01,
    max_len=200,
    min_len_per_dim=5,
    max_centroids=10,
    prob_const=0.0,
    reduce_num_constants=True,
    use_skeleton=False,
    prob_rand=0.0,
    max_trials=1,
    n_prediction_points=200,
    prediction_sigmas="1,2,4,8,16",
)


class Generator:
    def __init__(self, params=None):
        if params is None:
            params = Namespace(**DEFAULTS)
        self.generator = RandomFunctions(params, SPECIAL_WORDS)
        self.converter = ExpressionConverter()
        self.canon = StructureCanonicalizer()
        self.tokenizer = FormulaTokenizer()

    def sample_formula(self, rng=None):
        if rng is None:
            rng = np.random.RandomState()

        # === [新增] 重试循环 ===
        # 如果采样的公式定义域太窄（导致生成数据失败返回None），就放弃它，重新生成一个公式结构
        while True:
            # 1. 生成结构
            tree, input_dimension, _, _, _ = self.generator.generate_multi_dimensional_tree(rng)
            f = tree.nodes[0]
            org_tree = deepcopy(tree)

            # 2. 尝试生成数据
            tree_f, data_f = self.generator.generate_datapoints(
                rng=rng,
                tree=tree,
                input_distribution_type="gaussian",
                n_input_points=100,
                n_prediction_points=0,
                prediction_sigmas=[],
                input_dimension=input_dimension,
                n_centroids=1,
                max_trials=100  # 给每个公式 100 次尝试填满数据的机会
            )

            # 3. 检查是否成功
            # 如果 data_f 为 None，说明这个公式由于 nan/inf 太多被丢弃了
            # continue 重新进入 while 循环，生成新的 tree
            if data_f is None:
                # print(f"Formula {f} failed to generate valid data. Resampling structure...")
                continue

                # 4. 如果成功，跳出循环，继续处理
            x_f = data_f["fit"][0]
            y_f = data_f["fit"][1]
            break

            # === y_f 归一化 ===
        y_min = np.min(y_f)
        y_max = np.max(y_f)
        y_range = y_max - y_min

        if y_range > 1e-9:
            y_f = (y_f - y_min) / y_range
        else:
            y_f = np.zeros_like(y_f)

        f_expr = self.converter.convert(str(f))
        f_expr = self.canon.get_canonical_skeleton(f_expr)
        f_id = self.tokenizer.encode(f_expr)

        return x_f, y_f, f_expr, f, f_id, input_dimension, org_tree

    def resample_formula(self, tree, input_dimension, rng=None):
        if rng is None:
            rng = np.random.RandomState()
        tree_f, data_f = self.generator.generate_datapoints(
            rng=rng,
            tree=tree,  ###这里的tree本来就是f对应的tree
            input_distribution_type="gaussian",  ###用正态分布生成数据
            n_input_points=100,  ###样本量
            n_prediction_points=0,  ##这里不用管
            prediction_sigmas=[],  ##这里不用管
            input_dimension=input_dimension,  ##输入维度，即x的个数，这个应该要改的吧？
            n_centroids=1,  ##高斯分布的个数
            max_trials=400)  ##生成的数据出错后重新生成的最大次数
        x_f = data_f["fit"][0]
        y_f = data_f["fit"][1]

        # === [新增] y_f 归一化到 [0, 1] ===
        y_min = np.min(y_f)
        y_max = np.max(y_f)
        y_range = y_max - y_min

        if y_range > 1e-9:  # 防止除以零 (如果 y 是常数函数)
            y_f = (y_f - y_min) / y_range
        else:
            # 如果 y 是常数 (max == min)，归一化后通常设为 0 或 0.5
            # 这里设为 0，避免数值噪声
            y_f = np.zeros_like(y_f)

        return x_f, y_f

    def mutate_formula(self, f, input_dim, tree, edit, rng=None):
        if rng is None:
            rng = np.random.RandomState()

        # === [新增] 重试循环 ===
        max_mutate_attempts = 10
        for _ in range(max_mutate_attempts):
            # 1. 尝试突变
            g = perturb_tree_operators(self.generator, f, n=edit, rng=rng)
            if not self.generator._check_no_trivial_minmax(g):
                g = perturb_tree_operators(self.generator, f, n=edit + 2, rng=rng)

            # 暂时替换 tree 用于生成数据
            original_nodes = tree.nodes
            tree.nodes = [g]

            # 2. 尝试生成数据
            tree_g, data_g = self.generator.generate_datapoints(
                rng=rng,
                tree=tree,
                input_distribution_type="gaussian",
                n_input_points=100,
                n_prediction_points=0,
                prediction_sigmas=[],
                input_dimension=input_dim,
                n_centroids=1,
                max_trials=100
            )

            # 3. 检查结果
            if data_g is None:
                # 突变出的公式不好，还原 tree，重试下一次突变
                tree.nodes = original_nodes
                continue

            # 4. 成功
            x_g = data_g["fit"][0]
            y_g = data_g["fit"][1]

            # y 归一化 (记得加)
            y_min = np.min(y_g)
            y_max = np.max(y_g)
            y_range = y_max - y_min
            if y_range > 1e-9:
                y_g = (y_g - y_min) / y_range
            else:
                y_g = np.zeros_like(y_g)

            g_expr = self.converter.convert(str(g))
            g_expr = self.canon.get_canonical_skeleton(g_expr)
            g_id = self.tokenizer.encode(g_expr)

            return x_g, y_g, g_expr, g, g_id, input_dim, tree

        # 如果尝试了多次突变都失败，退化为重新采样一个全新的公式 (Fallback)
        # print("Mutation failed too many times, sampling new formula.")
        return self.sample_formula(rng)


############################################################使用示例########################################################
if __name__ == "__main__":

    generator = Generator()
    print(generator.sample_formula())
