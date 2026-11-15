<a id="building-the-cuda-backend-on-glibc-241-debian-trixie-and-newer"></a>
# Сборка бэкэнда CUDA на glibc ≥ 2.41 (Debian trixie и новее)

**Автор:** Питер Грум · 18 июля 2026 г.

Признак: `make glm CUDA=1` выходит из строя с четырьмя ошибками, например

    mathcalls.h(79): ошибка: спецификация исключения несовместима со спецификацией
    предыдущая функция "cospi" (объявлена в ... crt/math_functions.h)

Причина: в glibc 2.41 добавлен C23 `sinpi/cospi/sinpif/cospif` с конфликтующими спецификациями исключений.
с объявлениями `math_functions.h` CUDA ≤ 12.9. **Заголовки CUDA 13.x исправляют эту проблему.**

Рецепт, который не требует ни наборов инструментов, принадлежащих root, ни пакетов драйверов дистрибутива (безопасен рядом с
`.run`-установленный драйвер или драйвер dkms — важно в настройках LXC, где пользовательское пространство должно соответствовать
версия модуля ядра хоста):

```bash
# micromamba (single static binary) + NVIDIA's conda channel = complete nvcc
curl -Ls https://micro.mamba.pm/api/micromamba/linux-64/latest | tar -xj -C /opt bin/micromamba
/opt/bin/micromamba create -y -p /opt/cuda13 -c nvidia -c conda-forge cuda-nvcc=13 cuda-cudart-dev=13
ln -s /opt/cuda13/lib /opt/cuda13/lib64      # Makefile expects lib64; conda ships lib

make glm CUDA=1 CUDA_ARCH=sm_86 CUDA_HOME=/opt/cuda13   # your arch here
```

Бинарные ссылки `libcudart.so.13` динамически — отправляйте их рядом с двоичными файлами и запускайте с помощью
`LD_LIBRARY_PATH` или вставьте его в `/usr/local/lib`. Драйвер ≥ минимум времени выполнения
только требование к хосту (любой последний драйвер запускает двоичные файлы CUDA 13).

Тупики, для справки: `nvidia-cuda-toolkit` Debian жестко зависит от драйвера Debian.
пользовательское пространство (версия конфликтует с драйвером `.run`/dkms — и его `nvidia-installer-cleanup`
postinst попытается удалить установленную вами вручную область пользователя); пипс `nvidia-cuda-nvcc-cu12`
Wheel поставляется только с `ptxas`, а не с интерфейсом компилятора.