
import matplotlib.pyplot as plt
import numpy as np
from keras import backend as K
from music21 import note, stream, duration, tempo
import tensorflow as tf
from keras.models import Model
from keras.layers import Input,Flatten, Dense, Conv2DTranspose, Reshape, Lambda, Activation, BatchNormalization, LeakyReLU, Dropout, ZeroPadding2D, UpSampling2D, Reshape, Permute, RepeatVector, Concatenate, Conv3D
from keras.layers import ReLU, Activation, LeakyReLU
from keras.callbacks import Callback

class WGAN_GP(Model):

    def __init__(self, input_dim,grad_weight,
     z_dim, batch_size, n_tracks,
    n_bars, n_steps_per_bar, n_pitches):

        super(WGAN_GP, self).__init__()
        #размер входного тензора
        self.input_dim = input_dim
        #вес для штрафа за градиент
        self.gp_weight = grad_weight
        #размерность пространства векторов для генератора
        self.z_dim = z_dim
        #размер батча
        self.batch_size = batch_size
        #количество дорожек(инструментов)
        self.n_tracks = n_tracks
        #количество тактов для генерации
        self.n_bars = n_bars
        #количество временных шагов в такте
        self.n_steps_per_bar = n_steps_per_bar
        #количество используемых нот
        self.n_pitches = n_pitches
        #потери генератора и дискриминатора
        self.d_losses = []
        self.g_losses = []
        #номер эпохи(может следует убрать)
        self.epoch = 0
        #количество шагов критика
        self.n_critic_steps = 5

        #перегрузка метода compile, для задания оптимизаторов и потерь сетей     
    def compile(self, d_optimizer, g_optimizer, d_loss_fn, g_loss_fn):
        super(WGAN_GP, self).compile()
        self.d_optimizer = d_optimizer
        self.g_optimizer = g_optimizer
        self.d_loss_fn = d_loss_fn
        self.g_loss_fn = g_loss_fn

        self.generator = self.create_generator()
        print(self.generator)
        self.critic = self.create_critic()
    #функция для подсчета штрафа за градиент
    def gradient_penalty(self,batch_size, real_images, fake_images):
            alpha = tf.random.normal([batch_size,1, 1, 1, 1], 0.0, 1.0)
            diff = fake_images - real_images
            interpolated = real_images + alpha * diff

            with tf.GradientTape() as gp_tape:
                gp_tape.watch(interpolated)
                pred = self.critic(interpolated, training = True)

            grads = gp_tape.gradient(pred, [interpolated])[0]
            norm = tf.sqrt(tf.reduce_sum(tf.square(grads), axis=[1, 2, 3, 4]))
            gp = tf.reduce_mean((norm - 1.0) ** 2)
            return gp
    # прописываем train_step
    def train_step(self, real_images):
        if isinstance(real_images, tuple):
            real_images = real_images[0]
        #размер входного батча
        batch_size = tf.shape(real_images)[0]
        #обучение критика
        for i in range(self.n_critic_steps):
            chords_noise = tf.random.normal( shape=(batch_size, self.z_dim))
            style_noise = tf.random.normal( shape=(batch_size, self.z_dim))
            melody_noise = tf.random.normal( shape=(batch_size, self.n_tracks, self.z_dim))
            groove_noise = tf.random.normal( shape=(batch_size, self.n_tracks, self.z_dim))
            #тензоры едениц и нулей для критика соответсвенно
            valid = tf.ones(shape=(batch_size,1), dtype=np.float32)
            fake = -tf.ones(shape=(batch_size,1), dtype=np.float32)

            #подстчет потерь
            with tf.GradientTape() as tape:
                fake_images = self.generator([chords_noise,style_noise,melody_noise,groove_noise], training= True)
                fake_logits = self.critic(fake_images, training = True)
                real_logits = self.critic(real_images, training = True)


                d_cost_real = self.d_loss_fn(real_logits, valid)
                d_cost_fake = self.d_loss_fn(fake_logits, fake)
                gp = self.gradient_penalty(batch_size, real_images, fake_images)

                d_loss = d_cost_real + d_cost_fake + gp * self.gp_weight
            d_gradient = tape.gradient(d_loss, self.critic.trainable_variables)

            self.d_optimizer.apply_gradients(
                zip(d_gradient, self.critic.trainable_variables)
            )

        #обучение генератора
        chords_noise = tf.random.normal( shape=(batch_size, self.z_dim))
        style_noise = tf.random.normal( shape=(batch_size, self.z_dim))
        melody_noise = tf.random.normal( shape=(batch_size, self.n_tracks, self.z_dim))
        groove_noise = tf.random.normal( shape=(batch_size, self.n_tracks, self.z_dim))
        #вектор едениц для обучения генератора
        valid = tf.ones(shape=(batch_size,1), dtype=np.float32)
        with tf.GradientTape() as tape:
            generated_images = self.generator([chords_noise,style_noise,melody_noise,groove_noise], training = True)

            gen_img_logits = self.critic(generated_images, training = True)#False mb
            g_loss = self.g_loss_fn(gen_img_logits,valid)

        gen_gradient = tape.gradient(g_loss, self.generator.trainable_variables)
        self.g_optimizer.apply_gradients(
            zip(gen_gradient, self.generator.trainable_variables)
        )
        #запоминаем потери
        self.d_losses.append(d_loss)
        self.g_losses.append(g_loss)
        return {"d_loss": d_loss, 'r': d_cost_real, 'f': d_cost_fake, 'gp': gp, "g_loss": g_loss}
    #промежуточная сеть для аккордов и мелодии
    def Temp_network(self):
        input_layer = Input(shape=(self.z_dim,),name='temporal_input')
        x = Reshape([1,1,self.z_dim])(input_layer)
        x = Conv2DTranspose(filters = 1024, kernel_size=(2,1))(x)
        x = BatchNormalization(momentum = 0.9)(x)
        x = ReLU()(x)
        x = Conv2DTranspose(filters = 32, kernel_size=(self.n_bars-1,1))(x)
        x = BatchNormalization(momentum = 0.9)(x)
        x = ReLU()(x)
        output_layer = Reshape([2,32])(x)
        return Model(input_layer, output_layer)
    #генератор тактов 
    def BarGenerator(self):
        input_layer = Input(shape=(self.z_dim*4,),name='bargen_input')
        x = Dense(1024)(input_layer)
        x = BatchNormalization(momentum = 0.9)(x)
        x = ReLU()(x)
        x = Reshape([2,1,512])(x)
        x = Conv2DTranspose(filters = 512, kernel_size=(2,1),strides=(2,1),padding = 'same')(x)
        x = BatchNormalization(momentum = 0.9)(x)
        x = ReLU()(x)
        x = Conv2DTranspose(filters = 256, kernel_size=(2,1),strides=(2,1),padding = 'same')(x)
        x = BatchNormalization(momentum = 0.9)(x)
        x = ReLU()(x)
        x = Conv2DTranspose(filters = 256, kernel_size=(2,1),strides=(2,1),padding = 'same')(x)
        x = BatchNormalization(momentum = 0.9)(x)
        x = ReLU()(x)
        x = Conv2DTranspose(filters = 256, kernel_size=(1,7),strides=(1,7),padding = 'same')(x)
        x = BatchNormalization(momentum = 0.9)(x)
        x = ReLU()(x)
        x = Conv2DTranspose(filters = 1, kernel_size=(1,12),strides=(1,12),padding = 'same')(x)
        x = Activation(activation='tanh')(x)
        output_layer = Reshape([1, self.n_steps_per_bar , self.n_pitches ,1])(x)
        return Model(input_layer,output_layer)

    def create_generator(self):
        chords_input = Input(shape=(self.z_dim,), name='chords_input')
        style_input = Input(shape=(self.z_dim,), name='style_input')
        melody_input = Input(shape=(self.n_tracks, self.z_dim), name='melody_input')
        groove_input = Input(shape=(self.n_tracks, self.z_dim), name='groove_input')
        
        #Промежуточная сеть для аккордов
        chords_temp = self.Temp_network()
        chords_over_time = chords_temp(chords_input)
        
        #Промежуточная сеть для мелодии
        melody_over_time = [None] * self.n_tracks
        melody_tempNetwork = [None] * self.n_tracks
        for track in range(self.n_tracks):
            melody_tempNetwork[track] = self.Temp_network()
            melody_track = Lambda(lambda x: x[:,track,:])(melody_input)
            melody_over_time[track] = melody_tempNetwork[track](melody_track)
        
        #Генератор тактов по каждой дорожке
        barGen = [None] * self.n_tracks
        for track in range(self.n_tracks):
            barGen[track] = self.BarGenerator()
        
        bars_output = [None] * self.n_bars
        for bar in range(self.n_bars):
            track_output = [None] * self.n_tracks

            c = Lambda(lambda x: x[:,bar,:], name = 'chords_input_bar_' + str(bar))(chords_over_time)
            s = style_input

            for track in range(self.n_tracks):

                m = Lambda(lambda x: x[:,bar,:])(melody_over_time[track])
                g = Lambda(lambda x: x[:,track,:])(groove_input)
                
                z_input = Concatenate(axis = 1, name = 'total_input_bar_{}_track_{}'.format(bar, track))([c,s,m,g])
                
                track_output[track] = barGen[track](z_input)

            bars_output[bar] = Concatenate(axis = -1)(track_output)

        generator_output = Concatenate(axis = 1, name = 'concat_bars')(bars_output)

        return Model([chords_input, style_input, melody_input, groove_input], generator_output)
    #критик
    def create_critic(self):
        input_critic = Input(shape=self.input_dim,name='critic_input')
        x = Conv3D(filters=128, kernel_size=(2,1,1),strides=(1,1,1),padding='valid')(input_critic)
        x = LeakyReLU()(x)
        x = Conv3D(filters=128, kernel_size=(self.n_bars - 1,1,1),strides=(1,1,1),padding='valid')(x)
        x = LeakyReLU()(x)
        x = Conv3D(filters=128, kernel_size=(1,1,12),strides=(1,1,12),padding='same')(x)
        x = LeakyReLU()(x)
        x = Conv3D(filters=128, kernel_size=(1,1,7),strides=(1,1,7),padding='same')(x)
        x = LeakyReLU()(x)
        x = Conv3D(filters=128, kernel_size=(1,2,1),strides=(1,2,1),padding='same')(x)
        x = LeakyReLU()(x)
        x = Conv3D(filters=128, kernel_size=(1,2,1),strides=(1,2,1),padding='same')(x)
        x = LeakyReLU()(x)
        x = Conv3D(filters=256, kernel_size=(1,4,1),strides=(1,2,1),padding='same')(x)
        x = LeakyReLU()(x)
        x = Conv3D(filters=512, kernel_size=(1,3,1),strides=(1,2,1),padding='same')(x)
        x = LeakyReLU()(x)
        x = Flatten()(x)
        x = Dense(1024)(x)
        x = LeakyReLU()(x)
        output = Dense(1)(x)
        return Model(input_critic,output)
    #сохранение модели
    def save(self):
        self.critic.save('wgan/critic.h5')
        self.generator.save('wgan/generator.h5')
    #подгрузка модели
    def load(self, filepath_gen,filepath_disc):
        self.generator.load_weights(filepath_gen)
        self.critic.load_weights(filepath_disc)
    #построение графиков потерь
    def plot_loss(self):
        x = np.arange(1,len(self.d_losses) + 1)
        plt.plot(x,self.d_losses,label = 'discriminator loss')
        plt.plot(x,self.g_losses, label = 'generator loss')
        plt.xlabel('epochs')
        plt.ylabel('losses')
        plt.legend()

#callback handler для сохранения модели и сэмплов
class Monitor(Callback):
    def __init__(self,z_dim,n_bars, n_steps_per_bar, n_tracks, filepath):
        self.z_dim = z_dim
        self.filepath = filepath
        self.n_tracks = n_tracks
        self.n_bars = n_bars
        self.n_steps_per_bar = n_steps_per_bar

    def on_epoch_end(self,epoch, logs = None):
        if epoch %1000 == 0:
            chords_noise = tf.random.normal( shape=(1,self.z_dim))
            style_noise = tf.random.normal( shape=(1,self.z_dim))
            melody_noise = tf.random.normal( shape=(1,self.n_tracks, self.z_dim))
            groove_noise = tf.random.normal( shape=(1,self.n_tracks, self.z_dim))
            generated_image = self.model.generator.predict([chords_noise, style_noise,melody_noise, groove_noise])

            notes_to_midi(self.n_tracks,self.n_bars, self.n_steps_per_bar, epoch, generated_image, epoch)
#функция потерь Вассерштейна
def Wasserstein(y_true,y_pred):
    return -tf.reduce_mean(y_true*y_pred)
#из выхода генератора берем только максимальное значение для ноты
def binarise_output(output):
    max_pitches = np.argmax(output, axis = 3)
    return max_pitches
#функция для создания midi файлов
def notes_to_midi(n_tracks, n_bars, n_steps_per_bar, epoch, output, filename = None):

    for score_num in range(len(output)):

        max_pitches = binarise_output(output)
        midi_note_score = max_pitches[score_num].reshape([n_bars * n_steps_per_bar, n_tracks])
        parts = stream.Score()
        parts.append(tempo.MetronomeMark(number= 66))

        for i in range(n_tracks):
            last_x = int(midi_note_score[:,i][0])
            s= stream.Part()
            dur = 0
                

            for idx, x in enumerate(midi_note_score[:, i]):
                x = int(x)
                
                if (x != last_x or idx % 4 == 0) and idx > 0:
                    n = note.Note(last_x)
                    n.duration = duration.Duration(dur)
                    s.append(n)
                    dur = 0

                last_x = x
                dur = dur + 0.25
                
            n = note.Note(last_x)
            n.duration = duration.Duration(dur)
            s.append(n)
                
            parts.append(s)

        if filename is None:
            parts.write('midi', fp=("samples/sample_{}_{}.midi".format(epoch, score_num)))
        else:
            parts.write('midi', fp=("samples/{}.midi".format(filename)))
