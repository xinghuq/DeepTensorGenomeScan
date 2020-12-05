### python implementation based on GenNet (Hilten et al 2020)
import os
import sys
import warnings

import matplotlib

warnings.filterwarnings('ignore')
matplotlib.use('agg')
sys.path.insert(1, os.path.dirname(os.getcwd()))
import tensorflow 
import tensorflow.keras

tf.keras.backend.set_epsilon(0.0000001)

def evaluate_performance_regression(y, p):
    y = y.flatten()
    p = p.flatten()
    explained_variance = explained_variance_score(y, p)
    mse = mean_squared_error(y, p)
    r2 = r2_score(y, p)
    print("Mean squared error =", mse)
    print("Explained variance =", explained_variance)
    # print("maximum error =", maximum_error)
    print("r2 =", r2)

    plt.figure()
    df = pd.DataFrame([])
    df["truth"] = y
    df["predicted"] = p

    fig = sns.jointplot(x="truth", y="predicted", data=df, alpha=0.5)
    return fig, mse, explained_variance, r2
 def create_importance_csv(datapath, model, masks):
    network_csv = pd.read_csv(datapath + "/topology.csv")

    coordinate_list = []
    for i, mask in zip(np.arange(len(masks)), masks):
        coordinates = pd.DataFrame([])

        if (i == 0):
            if 'chr' in network_csv.columns:
                coordinates["chr"] = network_csv["chr"]
        coordinates["node_layer_" + str(i)] = mask.row
        coordinates["node_layer_" + str(i + 1)] = mask.col
        coordinates = coordinates.sort_values("node_layer_" + str(i), ascending=True)
        coordinates["weights_" + str(i)] = model.get_layer(name="LocallyDirected_" + str(i)).get_weights()[0]

        coordinate_names = network_csv[["layer" + str(i) + "_node", "layer" + str(i) + "_name"]].drop_duplicates()
        coordinate_names = coordinate_names.rename({"layer" + str(i) + "_node": "node_layer_" + str(i)}, axis=1)
        coordinates = coordinates.merge(coordinate_names, on="node_layer_" + str(i))
        coordinate_list.append(coordinates)

        if i == 0:
            total_list = coordinate_list[i]
        else:
            total_list = total_list.merge(coordinate_list[i], on="node_layer_" + str(i))

    i += 1
    coordinates = pd.DataFrame([])
    coordinates["weights_" + str(i)] = model.get_layer(name="output_layer").get_weights()[0].flatten()
    coordinates["node_layer_" + str(i)] = np.arange(len(coordinates))
    coordinate_names = network_csv[["layer" + str(i) + "_node", "layer" + str(i) + "_name"]].drop_duplicates()
    coordinate_names = coordinate_names.rename({"layer" + str(i) + "_node": "node_layer_" + str(i)}, axis=1)
    coordinates = coordinates.merge(coordinate_names, on="node_layer_" + str(i))
    total_list = total_list.merge(coordinates, on="node_layer_" + str(i))
    total_list["raw_importance"] = total_list.filter(like="weights").prod(axis=1)
    return total_list   
    
def train_regression(args):
    datapath = args.path
    jobid = args.ID
    lr_opt = args.learning_rate
    batch_size = args.batch_size
    epochs = args.epochs
    l1_value = args.L1
    problem_type = args.problem_type
    check_data(datapath, problem_type)

    optimizer_model = tf.keras.optimizers.Adam(lr=lr_opt)

    train_size = sum(pd.read_csv(datapath + "subjects.csv")["set"] == 1)
    val_size = sum(pd.read_csv(datapath + "subjects.csv")["set"] == 2)
    test_size = sum(pd.read_csv(datapath + "subjects.csv")["set"] == 3)

    folder, resultpath = get_paths(jobid)

    print("jobid =  " + str(jobid))
    print("folder = " + str(folder))
    print("batchsize = " + str(batch_size))
    print("lr = " + str(lr_opt))

    model, masks = create_network_from_csv(datapath=datapath, l1_value=l1_value, regression=True)
    model.compile(loss="mse", optimizer=optimizer_model,
                  metrics=["mse"])

    with open(resultpath + '/model_architecture.txt', 'w') as fh:
        model.summary(print_fn=lambda x: fh.write(x + '\n'))

    earlystop = tf.keras.callbacks.EarlyStopping(monitor='val_loss', min_delta=0.0001, patience=10, verbose=1, mode='auto',
                                          restore_best_weights=True)
    saveBestModel = tf.keras.callbacks.ModelCheckpoint(resultpath + "bestweights_job.h5", monitor='val_loss',
                                                verbose=1, save_best_only=True, mode='auto')

   
    if os.path.exists(resultpath + '/bestweights_job.h5'):
        print('Model already Trained')
    else:
        history = model.fit_generator(
            generator=traindata_generator(datapath=datapath,
                                          batch_size=batch_size,
                                          trainsize=int(train_size)),
            shuffle=True,
            epochs=epochs,
            verbose=1,
            callbacks=[earlystop, saveBestModel],
            workers=15,
            use_multiprocessing=True,
            validation_data=valdata_generator(datapath=datapath, batch_size=batch_size, valsize=val_size)
        )
        plt.plot(history.history['loss'])
        plt.plot(history.history['val_loss'])
        plt.title('model loss')
        plt.ylabel('loss')
        plt.xlabel('epoch')
        plt.legend(['train', 'val'], loc='upper left')
        plt.savefig(resultpath + "train_val_loss.png")
        plt.show()

    model.load_weights(resultpath + '/bestweights_job.h5')
    print("Finished")
    print("Analysis over the validation set")
    pval = model.predict_generator(
        valdata_generator(datapath=datapath, batch_size=1, valsize=val_size))
    yval = get_labels(datapath, set_number=2)
    fig, mse_val, explained_variance_val, r2_val = evaluate_performance_regression(yval, pval)
    np.save(resultpath + "/pval.npy", pval)
    fig.savefig(resultpath + "/validation_predictions.png", bbox_inches='tight', pad_inches=0)

    print("Analysis over the test set")
    ptest = model.predict_generator(
        testdata_generator(datapath=datapath, batch_size=1, testsize=test_size))
    ytest = get_labels(datapath, set_number=3)
    fig, mse_test, explained_variance_test, r2_test = evaluate_performance_regression(ytest, ptest)
    np.save(resultpath + "/ptest.npy", ptest)
    fig.savefig(resultpath + "/test_predictions.png", bbox_inches='tight', pad_inches=0)

    # %%

    with open(resultpath + '/Results_' + str(jobid) + '.txt', 'a') as f:
        f.write('\n Jobid = ' + str(jobid))
        f.write('\n Batchsize = ' + str(batch_size))
        f.write('\n Learningrate = ' + str(lr_opt))
        f.write('\n Optimizer = ' + str(optimizer_model))
        f.write('\n L1 value = ' + str(l1_value))
        f.write('\n')
        f.write("Validation set")
        f.write('\n Mean squared error = ' + str(mse_val))
        f.write('\n Explained variance = ' + str(explained_variance_val))
        # f.write('\n Maximum error = ' + str(maximum_error_val))
        f.write('\n R2 = ' + str(r2_val))
        f.write("Test set")
        f.write('\n Mean squared error = ' + str(mse_test))
        f.write('\n Explained variance = ' + str(explained_variance_val))
        # f.write('\n Maximum error = ' + str(maximum_error_test))
        f.write('\n R2 = ' + str(r2_test))
    importance_csv = create_importance_csv(datapath, model, masks)
    importance_csv.to_csv(resultpath + "connection_weights.csv")
