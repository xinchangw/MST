import numpy as np
import tensorflow as tf
from functools import reduce


def _num_weight_features(num_features, is_bias):
    return num_features - 1 + int(bool(is_bias))


def _reshape_inputs(A, num_features):
    A = np.asarray(A, dtype=np.float32)
    n = int(A.shape[1] / num_features)
    input_features = A.reshape((-1, num_features, n))
    return A, input_features, n


def _initial_weights(shape, fit_init):
    if fit_init is not None and hasattr(fit_init, "model_coef"):
        init_weights = np.asarray(fit_init.model_coef, dtype=np.float32)
        if init_weights.shape == shape:
            return init_weights.copy()
    return np.zeros(shape, dtype=np.float32)


def _broadcast_weights_tf(weights, model_type, n):
    if model_type == 0:
        return weights
    return tf.broadcast_to(weights, [1, tf.shape(weights)[1], n])


def _broadcast_weights_np(weights, model_type, n):
    if model_type == 0:
        return weights
    return np.broadcast_to(weights, (1, weights.shape[1], n))


def _mnl_logits_tf(input_features, weights, model_type, is_bias):
    n = tf.shape(input_features)[2]
    weight_scaled = _broadcast_weights_tf(weights, model_type, n)
    feature_block = input_features if is_bias else input_features[:, 1:, :]
    logits = tf.reduce_sum(feature_block * weight_scaled, axis=1)
    availability = input_features[:, 0, :]
    masked_logits = tf.where(
        availability > 0,
        logits,
        tf.fill(tf.shape(logits), tf.constant(-1.0e9, dtype=tf.float32)),
    )
    return logits, masked_logits


def _mnl_probabilities_np(input_features, weights, model_type, is_bias):
    n = input_features.shape[2]
    weight_scaled = _broadcast_weights_np(weights, model_type, n)
    feature_block = input_features if is_bias else input_features[:, 1:, :]
    logits = np.sum(feature_block * weight_scaled, axis=1)
    masked_logits = np.where(input_features[:, 0, :] > 0, logits, -1.0e9)
    masked_logits = masked_logits - np.max(masked_logits, axis=1, keepdims=True)
    probas = np.exp(masked_logits)
    probas = np.maximum(probas, 1.0e-5)
    probas = probas / np.sum(probas, axis=1, keepdims=True)
    return logits, probas


def _average_rank(probas, labels):
    ordered = np.argsort(-probas, axis=1)
    ranks = np.empty_like(ordered)
    rows = np.arange(ordered.shape[0])[:, None]
    ranks[rows, ordered] = np.arange(ordered.shape[1])
    return float(np.mean(ranks[np.arange(labels.shape[0]), labels]))


class LeafModelTensorflow(object):
    def __init__(self):
        return

    def fit(
        self,
        A,
        Y,
        weights,
        fit_init=None,
        refit=False,
        mode="mnl",
        batch_size=50,
        path="",
        model_type=0,
        num_features=2,
        epochs=10,
        steps=20000,
        steps_refit=60000,
        learning_rate=0.01,
        learning_rate_refit=0.001,
        is_bias=True,
        loglik_proba_cap=0,
        **kwargs
    ):
        self.loglik_proba_cap = loglik_proba_cap

        if mode != "mnl":
            raise NotImplementedError(
                "The TensorFlow leaf model currently supports only mode='mnl' "
                "after the TF2 migration."
            )

        if refit:
            steps = steps_refit
            learning_rate = learning_rate_refit

        A, input_features, n = _reshape_inputs(A, num_features)
        Y = np.asarray(Y, dtype=np.int32)
        weights = np.asarray(weights, dtype=np.float32)
        if A.shape[0] == 0:
            return 1

        weight_shape = (1, _num_weight_features(num_features, is_bias), n if model_type == 0 else 1)
        weights_var = tf.Variable(_initial_weights(weight_shape, fit_init), dtype=tf.float32, name="weights")
        optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate)

        dataset = tf.data.Dataset.from_tensor_slices((A, Y, weights))
        dataset = dataset.shuffle(min(A.shape[0], 10000), reshuffle_each_iteration=True).repeat()
        dataset = dataset.batch(max(1, min(batch_size, A.shape[0])), drop_remainder=False).prefetch(tf.data.AUTOTUNE)

        @tf.function
        def train_step(batch_x, batch_y, batch_w):
            with tf.GradientTape() as tape:
                batch_inputs = tf.reshape(batch_x, [-1, num_features, n])
                _, masked_logits = _mnl_logits_tf(batch_inputs, weights_var, model_type, is_bias)
                losses = tf.keras.losses.sparse_categorical_crossentropy(
                    batch_y,
                    masked_logits,
                    from_logits=True,
                )
                loss = tf.reduce_mean(losses * tf.cast(batch_w, tf.float32))
            gradients = tape.gradient(loss, [weights_var])
            optimizer.apply_gradients(zip(gradients, [weights_var]))
            return loss

        try:
            iterator = iter(dataset)
            for _ in range(int(steps)):
                batch_x, batch_y, batch_w = next(iterator)
                train_step(batch_x, batch_y, batch_w)
        except Exception:
            raise

        params_model = weights_var.numpy()
        if np.any(np.isnan(params_model)):
            return 1

        _, probas = _mnl_probabilities_np(input_features, params_model, model_type, is_bias)
        predictions = np.argmax(probas, axis=1)
        chosen = probas[np.arange(Y.shape[0]), Y]
        weighted_loss = float(np.mean(-np.log(np.maximum(chosen, 1.0e-9)) * weights))

        self.n = n
        self.num_features = num_features
        self.model_type = model_type
        self.is_bias = is_bias
        self.model_obj = weights_var
        self.model_coef = params_model
        self.model_quality = {
            "accuracy": float(np.mean(predictions == Y)),
            "average_rank": _average_rank(probas, Y),
            "loss": weighted_loss,
            "average_choice_probability": float(np.mean(chosen)),
        }
        return 0

    def predict(self, A, batch_size=50, name="probabilities", *args, **kwargs):
        _, input_features, _ = _reshape_inputs(A, self.num_features)
        _, pred_mat = _mnl_probabilities_np(
            input_features,
            self.model_coef,
            self.model_type,
            self.is_bias,
        )
        return pred_mat

    def error(self, A, Y):
        loglik_proba_cap = self.loglik_proba_cap
        Ypred = self.predict(A)
        log_probas = -np.log(np.maximum(Ypred[(np.arange(Y.shape[0]), Y)], loglik_proba_cap))
        return log_probas

    def error_pruning(self, A, Y):
        Ypred = self.predict(A)
        Z = np.zeros(Ypred.shape)
        Z[(np.arange(Y.shape[0]), Y)] = 1.0
        errors = np.sum((Z - Ypred) ** 2, axis=1)
        return errors

    def to_string(self, *leafargs, **leafkwargs):
        return "Model parameters: " + reduce(lambda x, y: x + "_" + str(y), self.model_coef, "")
    
#    #not needed to specify for other leaf models
#    def eval_model(self, A,Y, weights = None,batch_size = 100):
#        '''
#        Evaluates the model on a holdout dataset
#        '''
#        
#        eval_size = (A.shape[0]/batch_size)*batch_size
#        if eval_size < A.shape[0]:
#            c =  eval_size + batch_size - A.shape[0]
#            B = np.concatenate((A,A[:c,:]),axis = 0)
#            Y2 = np.concatenate((Y,Y[:c]),axis = 0)
#        else:
#            B = A
#            Y2 = Y
#        
#        if weights == None:
#            weights = np.ones(B.shape[0])        
#        eval_input_fn = tf.estimator.inputs.numpy_input_fn(
#                  x={"x": np.float32(B),
#                     "weight_data":np.float32(weights)},
#                  y=np.int32(Y2),
#                  num_epochs=1,
#                  batch_size=batch_size,
#                  shuffle=False)
#        eval_results = self.model_obj.evaluate(input_fn=eval_input_fn)
#        return(eval_results)
        

