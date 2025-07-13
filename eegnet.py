import numpy as np
import tensorflow as tf
from tensorflow.keras.layers import (Input, Conv2D, BatchNormalization, Activation,
                                     DepthwiseConv2D, SeparableConv2D, AveragePooling2D,
                                     Dropout, Flatten, Dense, Reshape)
from tensorflow.keras.constraints import max_norm
from tensorflow.keras.optimizers import Adam
import matplotlib.pyplot as plt
from scipy.io import loadmat

from features import Features
from csp_featues import CspFeatures

def EEGNet(input_shape, num_classes=4, sampling_rate=128, F1=8, D=2, F2=16,
           dropout_rate=0.5, kernel_length=64, norm_rate=0.25):
    """
    EEGNet模型实现 (兼容TF 2.17)
    参数:
        input_shape: 输入数据的形状 (channels, time_points)
        num_classes: 分类数量
        sampling_rate: 采样率 (Hz)
        F1: 第一层卷积核数量
        D: 深度乘数 (Depth multiplier)
        F2: 第二层卷积核数量
        dropout_rate: Dropout比例
        kernel_length: 时间卷积核长度 (建议设为采样率的一半)
        norm_rate: 权重约束的最大范数
    """
    # 添加通道维度
    input_layer = Input(shape=input_shape)

    # 重塑为 (samples, channels, time_points, 1)
    x = Reshape((input_shape[0], input_shape[1], 1))(input_layer)

    # Block 1: 时间卷积
    x = Conv2D(F1, (1, kernel_length), padding='same',
               use_bias=False, kernel_constraint=max_norm(norm_rate))(x)
    x = BatchNormalization(axis=1)(x)

    # 深度卷积 (空间滤波)
    x = DepthwiseConv2D((input_shape[0], 1), use_bias=False,
                        depth_multiplier=D,
                        depthwise_constraint=max_norm(norm_rate))(x)
    x = BatchNormalization(axis=1)(x)
    x = Activation('elu')(x)

    # 平均池化
    x = AveragePooling2D((1, 4))(x)
    x = Dropout(dropout_rate)(x)

    # Block 2: 可分离卷积
    x = SeparableConv2D(
        F2, (1, 16),
        padding='same',
        use_bias=False,
        depthwise_constraint=max_norm(norm_rate),
        pointwise_constraint=max_norm(norm_rate)
    )(x)
    x = BatchNormalization(axis=1)(x)
    x = Activation('elu')(x)

    # 平均池化
    x = AveragePooling2D((1, 8))(x)
    x = Dropout(dropout_rate)(x)

    # 分类层
    x = Flatten()(x)
    x = Dense(num_classes, activation='softmax',
              kernel_constraint=max_norm(norm_rate))(x)

    return tf.keras.Model(inputs=input_layer, outputs=x)

def prepare_data(X, y, test_size=0.2, random_state=42):
    """
    准备训练和验证数据
    """
    # 转换为float32并标准化
    X = X.astype('float32')

    # 计算每个样本的均值和标准差
    sample_means = np.mean(X, axis=(1, 2), keepdims=True)
    sample_stds = np.std(X, axis=(1, 2), keepdims=True)

    # 标准化数据 (避免除以零)
    X = (X - sample_means) / (sample_stds + 1e-8)

    # 拆分训练集和验证集
    from sklearn.model_selection import train_test_split
    X_train, X_val, y_train, y_val = train_test_split(
        X, y, test_size=test_size, stratify=y, random_state=random_state
    )

    return X_train, X_val, y_train, y_val

def train_eegnet(X_train, y_train, X_val, y_val, input_shape,
                 epochs=300, batch_size=32, learning_rate=0.001):
    """
    训练EEGNet模型 (兼容TF 2.17)
    """
    # 创建模型
    model = EEGNet(input_shape, num_classes=4, sampling_rate=128,
                   F1=8, D=2, F2=16, dropout_rate=0.5, kernel_length=64)

    # 编译模型
    model.compile(
        optimizer=Adam(learning_rate=learning_rate),
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy']
    )

    # 自定义回调系统 - 修复学习率属性问题
    class CustomCallback:
        def __init__(self, patience=30, min_lr=1e-6, lr_factor=0.5, lr_patience=10):
            self.patience = patience
            self.min_lr = min_lr
            self.lr_factor = lr_factor
            self.lr_patience = lr_patience
            self.best_weights = None
            self.best_val_loss = float('inf')
            self.wait = 0
            self.lr_wait = 0
            self.current_lr = learning_rate
            self.stopped_epoch = 0

        def on_epoch_end(self, epoch, logs, model):
            current_val_loss = logs['val_loss']

            # 检查是否是最佳模型
            if current_val_loss < self.best_val_loss:
                self.best_val_loss = current_val_loss
                self.best_weights = model.get_weights()
                self.wait = 0
                self.lr_wait = 0
            else:
                self.wait += 1
                self.lr_wait += 1

                # 检查学习率衰减
                if self.lr_wait >= self.lr_patience:
                    # 计算新学习率
                    new_lr = max(self.min_lr, self.current_lr * self.lr_factor)

                    # 获取当前优化器配置
                    optimizer_config = model.optimizer.get_config()

                    # 更新学习率
                    optimizer_config['learning_rate'] = new_lr

                    # 创建新的优化器实例
                    new_optimizer = Adam.from_config(optimizer_config)

                    # 替换模型的优化器
                    model.optimizer = new_optimizer

                    # 更新当前学习率
                    self.current_lr = new_lr

                    print(f"学习率衰减至: {self.current_lr:.6f}")
                    self.lr_wait = 0

                # 检查早停
                if self.wait >= self.patience:
                    print(f"\n早停: 验证损失未改善超过 {self.patience} 轮")
                    model.stop_training = True
                    self.stopped_epoch = epoch
                    model.set_weights(self.best_weights)

    # 创建回调实例
    custom_cb = CustomCallback(patience=30)

    # 训练历史记录
    history = {'loss': [], 'val_loss': [], 'accuracy': [], 'val_accuracy': []}

    # 训练模型
    print("开始训练EEGNet模型...")
    print(f"训练样本数: {len(X_train)}, 验证样本数: {len(X_val)}")
    print(f"批量大小: {batch_size}, 初始学习率: {learning_rate}")

    for epoch in range(epochs):
        print(f"\n轮次 {epoch+1}/{epochs}")

        # 训练步骤
        epoch_loss = []
        epoch_acc = []

        # 打乱训练数据
        indices = np.arange(len(X_train))
        np.random.shuffle(indices)
        X_train_shuffled = X_train[indices]
        y_train_shuffled = y_train[indices]

        for i in range(0, len(X_train_shuffled), batch_size):
            batch_end = min(i + batch_size, len(X_train_shuffled))
            batch_X = X_train_shuffled[i:batch_end]
            batch_y = y_train_shuffled[i:batch_end]

            with tf.GradientTape() as tape:
                predictions = model(batch_X, training=True)
                loss = tf.keras.losses.sparse_categorical_crossentropy(batch_y, predictions)
                loss = tf.reduce_mean(loss)

            grads = tape.gradient(loss, model.trainable_variables)
            model.optimizer.apply_gradients(zip(grads, model.trainable_variables))

            # 计算准确率
            acc = tf.reduce_mean(tf.keras.metrics.sparse_categorical_accuracy(batch_y, predictions))

            epoch_loss.append(loss.numpy())
            epoch_acc.append(acc.numpy())

        # 计算平均训练损失和准确率
        train_loss = np.mean(epoch_loss)
        train_acc = np.mean(epoch_acc)

        # 验证步骤
        val_loss = []
        val_acc = []
        for i in range(0, len(X_val), batch_size):
            batch_end = min(i + batch_size, len(X_val))
            batch_X = X_val[i:batch_end]
            batch_y = y_val[i:batch_end]

            predictions = model(batch_X, training=False)
            loss = tf.reduce_mean(tf.keras.losses.sparse_categorical_crossentropy(batch_y, predictions))
            acc = tf.reduce_mean(tf.keras.metrics.sparse_categorical_accuracy(batch_y, predictions))

            val_loss.append(loss.numpy())
            val_acc.append(acc.numpy())

        val_loss = np.mean(val_loss)
        val_acc = np.mean(val_acc)

        # 记录历史
        history['loss'].append(train_loss)
        history['accuracy'].append(train_acc)
        history['val_loss'].append(val_loss)
        history['val_accuracy'].append(val_acc)

        print(f"训练损失: {train_loss:.4f}, 训练准确率: {train_acc:.4f}")
        print(f"验证损失: {val_loss:.4f}, 验证准确率: {val_acc:.4f}")
        print(f"当前学习率: {custom_cb.current_lr:.6f}")

        # 回调函数
        logs = {'val_loss': val_loss}
        custom_cb.on_epoch_end(epoch, logs, model)

        # 检查是否早停
        if hasattr(model, 'stop_training') and model.stop_training:
            print(f"训练在轮次 {custom_cb.stopped_epoch+1} 因早停而终止")
            break

    # 恢复最佳权重
    if custom_cb.best_weights is not None:
        model.set_weights(custom_cb.best_weights)

    return model, history

def plot_training_history(history):
    """
    绘制训练历史
    """
    if not history:
        print("无训练历史可绘制")
        return

    plt.figure(figsize=(12, 5))

    # 准确率曲线
    plt.subplot(1, 2, 1)
    if 'accuracy' in history and 'val_accuracy' in history:
        plt.plot(history['accuracy'], label='训练准确率')
        plt.plot(history['val_accuracy'], label='验证准确率')
        plt.title('准确率曲线')
        plt.ylabel('准确率')
        plt.xlabel('训练轮次')
        plt.legend()
        plt.grid(True, linestyle='--', alpha=0.7)
    else:
        print("缺少准确率数据")

    # 损失曲线
    plt.subplot(1, 2, 2)
    if 'loss' in history and 'val_loss' in history:
        plt.plot(history['loss'], label='训练损失')
        plt.plot(history['val_loss'], label='验证损失')
        plt.title('损失曲线')
        plt.ylabel('损失')
        plt.xlabel('训练轮次')
        plt.legend()
        plt.grid(True, linestyle='--', alpha=0.7)
    else:
        print("缺少损失数据")

    plt.tight_layout()
    plt.show()

def evaluate_model(model, X_test, y_test, batch_size=32):
    """
    评估模型性能
    """
    if model is None:
        print("模型未定义")
        return 0.0

    # 评估测试集
    test_loss = []
    test_acc = []
    y_pred = []

    for i in range(0, len(X_test), batch_size):
        batch_end = min(i + batch_size, len(X_test))
        batch_X = X_test[i:batch_end]
        batch_y = y_test[i:batch_end]

        predictions = model(batch_X, training=False)
        loss = tf.reduce_mean(tf.keras.losses.sparse_categorical_crossentropy(batch_y, predictions))
        acc = tf.reduce_mean(tf.keras.metrics.sparse_categorical_accuracy(batch_y, predictions))

        test_loss.append(loss.numpy())
        test_acc.append(acc.numpy())
        y_pred.extend(tf.argmax(predictions, axis=1).numpy())

    if test_loss:
        test_loss = np.mean(test_loss)
        test_acc = np.mean(test_acc)
    else:
        test_loss = 0.0
        test_acc = 0.0

    print(f"\n测试集准确率: {test_acc:.4f}")
    print(f"测试集损失: {test_loss:.4f}")

    # 混淆矩阵
    if y_pred:
        from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
        try:
            cm = confusion_matrix(y_test, y_pred)
            disp = ConfusionMatrixDisplay(
                confusion_matrix=cm,
                display_labels=['左手', '右手', '脚', '舌头']
            )
            disp.plot(cmap='Blues')
            plt.title('EEGNet分类混淆矩阵')
            plt.show()
        except Exception as e:
            print(f"绘制混淆矩阵时出错: {e}")

    return test_acc

# 主程序
if __name__ == "__main__":
    # 假设你已经加载了数据
    # X: EEG数据 (samples, channels, time_points)
    # y: 类别标签 (0-3)

    # 示例数据形状: (576, 22, 1251) - 假设采样率250Hz, 3秒试验
    # 实际使用你的数据加载代码
    cf = CspFeatures('erp', False)
    X_train, y_train = cf.get_train()

    # 设置输入形状 (通道数, 时间点数)
    input_shape = (X_train.shape[1], X_train.shape[2])

    X_test = []
    y_test = []

    # Predict
    for i in cf.tests:
        X_test.append(cf.get_test(Features(i, True)))
        e1_labels = loadmat(f'../data/BCICIV_2a_gdf/true_labels/A0{i}E.mat')
        y_test.append(e1_labels['classlabel'].reshape(288) - 1)

    X_test = np.vstack(X_test)
    y_test = np.hstack(y_test)

    # 训练模型
    model, history = train_eegnet(
        X_train, y_train,
        X_test, y_test,
        input_shape,
        epochs=300,
        batch_size=32,
        learning_rate=0.001
    )

    # 可视化训练过程
    plot_training_history(history)

    # 评估模型
    test_acc = evaluate_model(model, X_test, y_test)

    # 保存模型
    model.save('eegnet_bci_iv_2a.h5')
    print("模型已保存为 'eegnet_bci_iv_2a.h5'")