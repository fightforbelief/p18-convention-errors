import sys
sys.path.insert(0, __import__("os").path.join(__import__("os").path.dirname(__import__("os").path.abspath(__file__)), ".."))
import ablation as A


def build_panel(calls, attrs):
    return A.variant(calls, attrs, G=1, S=1, L=16)
