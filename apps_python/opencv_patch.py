"""OpenCV compatibility patch"""
import sys

# Patch cv2.dnn.DictValue before any import
import cv2
if not hasattr(cv2.dnn, 'DictValue'):
    # Create a dummy DictValue class
    class DictValue:
        pass
    cv2.dnn.DictValue = DictValue
    print("✓ Patched cv2.dnn.DictValue")

# Also patch the typing module if needed
try:
    from cv2.typing import LayerId
except (ImportError, AttributeError):
    # Define LayerId as int if not available
    import cv2.typing
    cv2.typing.LayerId = int
    print("✓ Patched cv2.typing.LayerId")
