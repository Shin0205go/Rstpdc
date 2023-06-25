const tf = require('@tensorflow/tfjs');
require('@tensorflow/tfjs-node');

let coffeeData = [];
let beanTypes = ["Arabica", "Robusta", "Liberica"];

for (let i = 0; i < 10000; i++) {
  let beanType = beanTypes[Math.floor(Math.random() * beanTypes.length)];
  let roastTime = getRandomArbitrary(5, 15);
  let temperature = getRandomArbitrary(190, 230);
  let humidity = getRandomArbitrary(0.3, 0.7);
  let initialHumidity = getRandomArbitrary(0.1, 0.2);
  let initialTemperature = getRandomArbitrary(15, 25);
  let quality = getRandomArbitrary(0.5, 1);

  coffeeData.push({ roastTime, temperature, humidity, beanType, initialHumidity, initialTemperature, quality });
}

coffeeData = oneHotEncoding(coffeeData, "beanType");
console.log(coffeeData);

let featuresToNormalize = ["roastTime", "temperature", "humidity", "initialHumidity", "initialTemperature", "quality"];
featuresToNormalize.forEach(feature => {
  coffeeData = normalize(coffeeData, feature);
});

let shuffledData = coffeeData.sort(() => 0.5 - Math.random());
let trainData = shuffledData.slice(0, Math.round(shuffledData.length * 0.8));
let testData = shuffledData.slice(Math.round(shuffledData.length * 0.8));

const inputTensor = tf.tensor2d(trainData.map(item => [item.roastTime, item.temperature, item.humidity, item.initialHumidity, item.initialTemperature]));
const targetTensor = tf.tensor2d(trainData.map(item => [item.quality]));

const model = tf.sequential();
model.add(tf.layers.dense({ units: 64, activation: 'relu', inputShape: [5] }));
model.add(tf.layers.dense({ units: 64, activation: 'relu' }));
model.add(tf.layers.dense({ units: 1, activation: 'linear' }));

model.compile({ optimizer: 'adam', loss: 'meanSquaredError' });


function getRandomArbitrary(min, max) {
  return Math.random() * (max - min) + min;
}

function oneHotEncoding(data, variable) {
  let uniqueValues = [...new Set(data.map(item => item[variable]))];
  for (let i = 0; i < data.length; i++) {
    for (let j = 0; j < uniqueValues.length; j++) {
      if (data[i][variable] === uniqueValues[j]) {
        data[i][`${variable}_${uniqueValues[j]}`] = 1;
      } else {
        data[i][`${variable}_${uniqueValues[j]}`] = 0;
      }
    }
    delete data[i][variable];
  }
  return data;
}

function normalize(data, feature) {
  let featureValues = data.map(item => item[feature]);
  let min = Math.min(...featureValues);
  let max = Math.max(...featureValues);
  for (let i = 0; i < data.length; i++) {
    data[i][feature] = (data[i][feature] - min) / (max - min);
  }
  return data;
}

async function trainModel() {
  await model.fit(inputTensor, targetTensor, { epochs: 100, shuffle: true });
  console.log('Training completed.');

function simulateRoasting(roastTime, temperature, humidity, beanType, initialHumidity, initialTemperature) {
  // One-hot encoding for beanType
  let encodedBeanType = [];
  for (let i = 0; i < beanTypes.length; i++) {
    if (beanType === beanTypes[i]) {
      encodedBeanType.push(1);
    } else {
      encodedBeanType.push(0);
    }
  }

  // Normalize input features
  roastTime = (roastTime - 5) / (15 - 5);
  temperature = (temperature - 190) / (230 - 190);
  humidity = (humidity - 0.3) / (0.7 - 0.3);
  initialHumidity = (initialHumidity - 0.1) / (0.2 - 0.1);
  initialTemperature = (initialTemperature - 15) / (25 - 15);

// Create input tensor for prediction
const input = tf.tensor2d([[roastTime, temperature, humidity, initialHumidity, initialTemperature]], [1, 5]);

  // Predict the quality
  const prediction = model.predict(input);

  // Denormalize the predicted quality
  const quality = prediction.dataSync()[0] * (1 - 0.5) + 0.5;

  return quality;
}

// Example usage:
const roastedBean = {
  roastTime: 10,
  temperature: 200,
  humidity: 0.5,
  beanType: "Arabica",
  initialHumidity: 0.15,
  initialTemperature: 20
};

tf.tidy(() => {
  const predictedQuality = simulateRoasting(
    roastedBean.roastTime,
    roastedBean.temperature,
    roastedBean.humidity,
    roastedBean.beanType,
    roastedBean.initialHumidity,
    roastedBean.initialTemperature
  );

  console.log("Predicted Quality:", predictedQuality);
});

}
trainModel();