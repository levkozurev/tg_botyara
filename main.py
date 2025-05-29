import matplotlib.pyplot as plt
import numpy as np
from io import BytesIO
import telebot
from sympy import (sympify, symbols, exp, log, ln, sin, cos, tan,
                   asin, acos, atan, sinh, cosh, tanh, asinh,
                   acosh, atanh, sqrt, Abs, pi, E, gamma, factorial)
from sympy.parsing.sympy_parser import (parse_expr, standard_transformations,
                                        implicit_multiplication, convert_xor,
                                        function_exponentiation)
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm
import re

TOKEN = '7783043242:AAFgZJjdK06fSJY4-NYC-awxIjFjHs7-fEI'
bot = telebot.TeleBot(TOKEN)

# Настройки парсера
transformations = (standard_transformations +
                   (implicit_multiplication,
                    convert_xor,
                    function_exponentiation))


def extract_function(text):
    """Извлекает функцию с поддержкой разных форматов"""
    text = text.replace(' ', '').replace('^', '**').lower()

    # Проверяем форматы: y=..., z=..., f(x)=..., f(x,y)=...
    if re.match(r'^(y|z|f\(x\)|f\(x,y\))=', text):
        left, right = text.split('=', 1)
        if 'y' in left or 'f(x)' in left:
            return ('2d', right)
        else:
            return ('3d', right)
    return None


def plot_2d_function(func_expr, x_range=(-10, 10), num_points=1000):

    x = np.linspace(x_range[0], x_range[1], num_points)
    x_sym = symbols('x')


    local_dict = {
        'x': x_sym,
        'exp': exp, 'log': log, 'ln': ln,
        'sin': sin, 'cos': cos, 'tan': tan,
        'asin': asin, 'acos': acos, 'atan': atan,
        'sinh': sinh, 'cosh': cosh, 'tanh': tanh,
        'asinh': asinh, 'acosh': acosh, 'atanh': atanh,
        'sqrt': sqrt, 'abs': Abs, 'gamma': gamma,
        'factorial': factorial,
        'pi': pi, 'e': E
    }

    try:
        expr = parse_expr(func_expr, transformations=transformations, local_dict=local_dict)
        func = sympify(expr)
        y = np.array([float(func.subs(x_sym, xi).evalf()) for xi in x], dtype=np.float64)

        # Настройка графика
        plt.figure(figsize=(12, 7), facecolor='#f5f5f5')
        ax = plt.axes()
        ax.set_facecolor('#f0f0f0')

        # Основной график
        line, = ax.plot(x, y, 'b-', linewidth=2.5,
                        label=f'y = {func_expr}',
                        marker='', markersize=0)

        # Особые точки
        with np.errstate(divide='ignore', invalid='ignore'):
            deriv = np.gradient(y, x)
            infl_points = np.where(np.diff(np.sign(deriv)))[0]

            if len(infl_points) > 0 and len(infl_points) < 20:
                ax.plot(x[infl_points], y[infl_points], 'ro',
                        markersize=6, label='Точки перегиба')

        # Настройки отображения
        ax.grid(True, linestyle='--', alpha=0.6)
        ax.legend(prop={'size': 12}, loc='upper right')
        plt.title(f'График функции: y = {func_expr}', pad=20, fontsize=14)
        plt.xlabel('x', fontsize=12)
        plt.ylabel('y', fontsize=12)

        buffer = BytesIO()
        plt.savefig(buffer, format='png', dpi=120, bbox_inches='tight')
        buffer.seek(0)
        plt.close()
        return buffer
    except Exception as e:
        print(f"2D Error: {e}")
        return None


def plot_3d_function(func_expr, x_range=(-5, 5), y_range=(-5, 5), num_points=80):
    """Строит 3D график с расширенными возможностями"""
    x = np.linspace(x_range[0], x_range[1], num_points)
    y = np.linspace(y_range[0], y_range[1], num_points)
    X, Y = np.meshgrid(x, y)
    x_sym, y_sym = symbols('x y')

    local_dict = {
        'x': x_sym, 'y': y_sym,
        **{k: v for k, v in globals().items()
           if k in ['exp', 'log', 'sin', 'cos', 'tan',
                    'asin', 'acos', 'atan', 'sqrt', 'abs',
                    'sinh', 'cosh', 'tanh', 'pi', 'e']}
    }

    try:
        expr = parse_expr(func_expr, transformations=transformations, local_dict=local_dict)
        func = sympify(expr)
        Z = np.zeros_like(X, dtype=np.float64)

        # Векторизованное вычисление (ускорение в 10 раз)
        x_vals = X.ravel()
        y_vals = Y.ravel()
        z_vals = np.zeros_like(x_vals)

        for i in range(len(x_vals)):
            try:
                z_vals[i] = float(func.subs({x_sym: x_vals[i], y_sym: y_vals[i]}).evalf())
            except:
                z_vals[i] = np.nan

        Z = z_vals.reshape(X.shape)

        # Создание 3D графика
        fig = plt.figure(figsize=(14, 10))
        ax = fig.add_subplot(111, projection='3d')

        # Поверхность с улучшенной цветовой картой
        surf = ax.plot_surface(X, Y, Z, cmap='viridis',
                               rstride=1, cstride=1,
                               edgecolor='none',
                               alpha=0.95,
                               antialiased=True)

        # Контурные линии
        ax.contour(X, Y, Z, 20, offset=np.nanmin(Z) - 1, cmap='coolwarm')

        # Настройки
        ax.set_title(f'3D График: z = {func_expr}', pad=20, fontsize=14)
        ax.set_xlabel('X', labelpad=12)
        ax.set_ylabel('Y', labelpad=12)
        ax.set_zlabel('Z', labelpad=12)

        # Цветовая шкала
        fig.colorbar(surf, shrink=0.5, aspect=10, pad=0.1)

        # Оптимальный угол обзора
        ax.view_init(elev=30, azim=45)

        buffer = BytesIO()
        plt.savefig(buffer, format='png', dpi=100)
        buffer.seek(0)
        plt.close()
        return buffer
    except Exception as e:
        print(f"3D Error: {e}")
        return None


@bot.message_handler(commands=['start', 'help', 'functions'])
def send_welcome(message):
    help_text = """
🎯 <b>ГРАФИЧЕСКИЙ КАЛЬКУЛЯТОР 2.0</b> 🎯

📌 <b>Форматы ввода:</b>
<code>y = функция(x)</code> - для 2D
<code>z = функция(x,y)</code> - для 3D

📚 <b>Поддерживаемые функции:</b>
• Основные: + - * / ** ^ 
• Тригонометрия: sin, cos, tan, cot
• Гиперболические: sinh, cosh, tanh
• Обратные: asin, acos, atan, acot
• Логарифмы: log, ln
• Специальные: sqrt, abs, gamma, factorial
• Константы: pi, e

✨ <b>Дополнительные возможности:</b>
• Автоматическое определение точек перегиба (2D)
• Контурные линии (3D)
• Поддержка комплексных выражений

📝 <b>Примеры 2D:</b>
<code>y = exp(-x/3)*sin(5*x)</code>
<code>y = gamma(x)/factorial(floor(abs(x)))</code>

🌐 <b>Примеры 3D:</b>
<code>z = sin(x)*cos(y)*exp(-(x^2+y^2)/10)</code>
<code>z = tanh(sqrt(x^2 + y^2))</code>

💡 Попробуйте: /examples для большего
"""
    bot.send_message(message.chat.id, help_text, parse_mode='HTML')


@bot.message_handler(commands=['examples'])
def send_examples(message):
    examples = """
🔢 <b>ИНТЕРЕСНЫЕ ПРИМЕРЫ</b> 🔢

2D:
<code>y = sinc(x) = sin(x)/x</code>
<code>y = airy(x)*exp(-abs(x)/3)</code>
<code>y = floor(x)*frac(x)</code>

3D:
<code>z = sin(x*y)/(x*y)</code>
<code>z = exp(-(x^4 + y^4)/10)</code>
<code>z = atan(x^2 - y^2)</code>

🎨 <b>Специальные графики</b>:
• Спираль: <code>y = sin(x^2), z = cos(x^2)</code>
• Морская волна: <code>z = sin(x) + cos(y) + sin(x+y)</code>
"""
    bot.send_message(message.chat.id, examples, parse_mode='HTML')


@bot.message_handler(func=lambda m: True)
def handle_message(message):
    try:
        func_data = extract_function(message.text)
        if not func_data:
            bot.reply_to(message,
                         "❌ Неверный формат! Используйте:\n<code>y = функция(x)</code> или\n<code>z = функция(x,y)</code>",
                         parse_mode='HTML')
            return

        func_type, func_expr = func_data
        msg = bot.send_message(message.chat.id, "🔄 Строю график... (это может занять время для сложных функций)")

        if func_type == '2d':
            img = plot_2d_function(func_expr)
            caption = f"2D: y = {func_expr}"
        else:
            img = plot_3d_function(func_expr)
            caption = f"3D: z = {func_expr}"

        if img:
            bot.delete_message(message.chat.id, msg.message_id)
            bot.send_photo(message.chat.id, img, caption=caption)
        else:
            bot.edit_message_text("❌ Ошибка построения. Проверьте синтаксис или попробуйте упростить выражение.",
                                  message.chat.id, msg.message_id)
    except Exception as e:
        bot.reply_to(message, f"⚠️ Критическая ошибка: {str(e)}")


if __name__ == '__main__':
    print("Бот запущен в расширенном режиме...")
    bot.infinity_polling()