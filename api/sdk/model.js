const tf = require('@tensorflow/tfjs-node');

function normalized(data){ // i & r
    a = (data[0] - 42.773) / 10.33017
    b = (data[1] - 29.9412) / 8.936247
    c = (data[2] - 94.8964) / 8.887377
    return [a, b, c]
}

function denormalized(data){
    x = (data[0] * 16.08198) + 32.2718
    y = (data[1] * 8.918185) + 29
    z = (data[2] * 11.79956) + 69.739
    return [x, y, z]
}


async function predict(data){
    let in_dim = 3;
    
    data = normalized(data);
    shape = [1, in_dim];

    tf_data = tf.tensor2d(data, shape);

    try{
        // path load in public access => github
        const path = 'https://raw.githubusercontent.com/RizkySyahry/bot-jst/main/public/JST_UAS/model%20(1).json';
        const model = await tf.loadGraphModel(path);
        
        predict = model.predict(
                tf_data
        );
        result = predict.dataSync();
        return denormalized( result );
        
    }catch(e){
      console.log(e);
    }
}

module.exports = {
    predict: predict 
}
  
