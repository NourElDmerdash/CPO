# Crested Porcupine Optimizer (CPO) - Python Implementation

هذا المشروع هو تنفيذ لخوارزمية Crested Porcupine Optimizer (CPO) بلغة Python، وهي خوارزمية تحسين جديدة مستوحاة من الطبيعة طورها Abdel-Basset, M. و Mohamed, R.

## نظرة عامة

خوارزمية CPO مستوحاة من آليات الدفاع لدى النيص الأفريقي ذو العرف (Crested Porcupine). تستخدم هذه الآليات للموازنة بين الاستكشاف والاستغلال في عملية البحث عن الحل الأمثل.

## هيكل المشروع

- `cpo.py` - التنفيذ الأساسي لخوارزمية CPO
- `benchmark_functions.py` - دوال مساعدة للدوال المعيارية CEC
- `cec_functions_wrapper.py` - واجهة ربط مع دوال CEC المنفذة بلغة C++
- `cec_functions_data.py` - إدارة ملفات بيانات CEC (مصفوفات الدوران، بيانات الإزاحة، الخ)
- `main.py` - سكريبت لتشغيل CPO على دوال CEC المعيارية
- `example.py` - مثال بسيط يوضح تشغيل CPO على دوال اختبار قياسية
- `testing.py` - اختبارات للتحقق من صحة واجهة دوال CEC

## المتطلبات

- NumPy (>= 1.19.0)
- Matplotlib (>= 3.3.0)
- SciPy (>= 1.5.0)
- CFFI (>= 1.14.0)
- مترجم C++ (مثل GCC) لترجمة كود C++ الخاص بدوال CEC

## طريقة الاستخدام

### تثبيت المتطلبات

```
pip install -r requirements.txt
```

### إعداد دوال CEC

1. تأكد من وجود ملفات C++ لدوال CEC في المجلد الرئيسي (مثل `cec17_func.cpp`)
2. تأكد من وجود ملفات البيانات اللازمة في مجلد `input_data17` للتوافق مع دوال CEC-2017

### اختبار دوال CEC

للتحقق من صحة واجهة دوال CEC:

```
python testing.py
```

### مثال بسيط

لتشغيل CPO على دوال اختبار معيارية بسيطة:

```
python example.py
```

### تشغيل على دوال CEC

لتشغيل CPO على دوال CEC المعيارية:

```
python main.py
```

## تنفيذ دوال CEC

يدعم المشروع الحالي دوال CEC لسنوات مختلفة:
- CEC-2014 (cec_year=1)
- CEC-2017 (cec_year=2)
- CEC-2020 (cec_year=3)
- CEC-2022 (cec_year=4)

الواجهة تعمل على إنشاء مكتبة مشتركة من ملفات C++ الموجودة، وبالتالي تحتاج إلى:

1. ملفات المصدر C++ (مثل `cec17_func.cpp`)
2. ملفات البيانات (مثل ملفات الإزاحة ومصفوفات الدوران) في المجلدات المناسبة

## استخدام CPO مع دالة تكلفة مخصصة

```python
import numpy as np
from cpo import cpo

# تعريف دالة التكلفة المخصصة
def my_objective_function(x):
    return np.sum(x**2)  # مثال: دالة Sphere

# تحديد أبعاد المشكلة والحدود
dim = 10
lb = -100 * np.ones(dim)
ub = 100 * np.ones(dim)

# تشغيل خوارزمية CPO
best_fitness, best_position, convergence = cpo(
    pop_size=50,
    max_iter=1000,
    ub=ub,
    lb=lb,
    dim=dim,
    obj_func=my_objective_function
)

print(f"أفضل لياقة: {best_fitness}")
print(f"أفضل موقع: {best_position}")
```

## المراجع

> Abdel-Basset, M., Mohamed, R. "Crested Porcupine Optimizer: A new nature-inspired metaheuristic" 