from collections import namedtuple

import tensorflow as tf


class AE(tf.keras.Model):

    def __init__(self, input_shape, **params):
        Parameters = namedtuple('Parameters', ['input_shape']+sorted(params))
        self.params = Parameters(input_shape=input_shape, **params)
        self.name = 'AE'

   
    def build(self):
        """ build full model: encoder + decoder"""
        inputs = tf.keras.layers.Input(shape=self.params.input_shape, dtype=tf.float32, name='model_input')
        self.encoder = self.build_encoder(inputs)
        self.latent = self.encoder(inputs)
        self.decoder = self.build_decoder(self.latent)
        # link encoder and decoder to full vae model
        outputs = self.decoder(latent)  # link encoder output to decoder
        # instantiate AE model
        self.model = tf.keras.Model(inputs, outputs, name=self.name)
        return self.model


    def build_encoder(self, inputs): # -> tf.keras.Model
        raise NotImplementedError


    def build_decoder(self, latent): # -> tf.keras.Model
        raise NotImplementedError


    def call(self, inputs):
        return self.model(inputs)


    def save(self, path):
        pathlib.Path(path).mkdir(parents=True, exist_ok=True)
        print('saving model to {}'.format(path))
        self.encoder.save(os.path.join(path, 'encoder.h5'))
        self.decoder.save(os.path.join(path, 'decoder.h5'))
        self.model.save(os.path.join(path, self.name+'.h5'))


    @classmethod
    def load(cls, path, custom_objects={}):
        ''' loading only for inference -> passing compile=False '''
        encoder = tf.keras.models.load_model(os.path.join(path, 'encoder.h5'), custom_objects=custom_objects, compile=False)
        decoder = tf.keras.models.load_model(os.path.join(path, 'decoder.h5'), custom_objects=custom_objects, compile=False)
        model = tf.keras.models.load_model(os.path.join(path, self.name+'.h5'), custom_objects=custom_objects, compile=False)
        return encoder, decoder, model


class VAE():

    def __init__(self, beta=1., **params):
        super().__init__(**params)
        self.name = 'VAE'


    @classmethod
    def from_saved_model(cls, path):
        encoder, decoder, model = cls.load(path)
        with h5py.File(os.path.join(path,'model_params.h5'),'r') as f: 
            params = f.get('params')
            beta = float(params.attrs['beta'])
        instance = cls(beta=beta)
        instance.encoder = encoder
        instance.decoder = decoder
        instance.model = model
        return instance

    @property
    def beta(self):
        return self.params.beta

    def fit(self, x, y, epochs=3, verbose=2):
        callbacks = [tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=7, verbose=1),tf.keras.callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.1, patience=2, verbose=1),tf.keras.callbacks.TerminateOnNaN(),
                     ] #TensorBoard(log_dir=self.log_dir, histogram_freq=1)
        self.history = self.model.fit(x, y, batch_size=self.params.batch_sz, epochs=epochs, verbose=verbose, callbacks=callbacks, validation_split=0.25)
        return self.history

    def predict(self, x):
        return self.model.predict(x, batch_size=1024)

    def predict_with_latent(self, x):
        z_mean, z_log_var, z = self.encoder.predict(x, batch_size=1024)
        reco = self.decoder.predict(z, batch_size=1024)
        return [reco, z_mean, z_log_var]

    def save(self, path):
        super().save(path)
        # sneak in beta factor as group attribute of vae.h5 file
        with h5py.File(os.path.join(path,'model_params.h5'),'w') as f:
            ds = f.create_group('params')
            ds.attrs['beta'] = self.params.beta

