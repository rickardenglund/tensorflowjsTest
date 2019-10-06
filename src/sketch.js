const xs = tf.tensor([
    [0, 0],
    [0, 1],
    [1, 0],
    [1, 1],
]);

const ys = tf.tensor([0, 1, 1, 0]);

const model = tf.sequential({
    layers: [
        tf.layers.dense({inputShape: [2], units: 16, activation: 'relu'}),
        tf.layers.dense({units: 1, activation: 'sigmoid'}),
    ]
});

const resolution = 50;
let testdata;
let frameRateParagraph;

function setup() {
    createCanvas(900, 900);
    frameRateParagraph = createP('fps: ');
    setInterval(() => {frameRateParagraph.elt.innerText = Math.floor(frameRate())}, 1000)

    model.compile({
        optimizer: tf.train.adam(0.1),
        loss: 'meanSquaredError',
        metrics: ['accuracy']
    });

    train(1);

    let values = [];
    for (let x = 0; x < width; x += resolution) {
        for (let y = 0; y < width; y += resolution) {
            values.push([
                map(x, 0, width, 0, 1),
                map(y, 0, width, 0, 1)
            ])
        }
    }
    testdata = tf.tensor(values)
}

function draw() {
    let vs = model.predict(testdata).dataSync();
    stroke(0,0,0, 50);

    let i = 0;
    for (let x = 0; x < width; x += resolution) {
        for (let y = 0; y < width; y += resolution) {
            fill(map(vs[i], 0, 1, 0, 255));
            rect(y, x, resolution, resolution);
            i++
        }
    }
}

function train(i) {
    model.fit(xs, ys, {
        shuffle: true,
        verbose: true,
        epochs: 10
    }).then(r => {
        console.log('Iteration: ' + i);
        console.log('loss: ' + r.history.loss[0]);
        console.log('acc: ' + r.history.acc[0]);

        if (r.history.acc[0] < 0.9999
            || r.history.loss[0] > 0.001) {
            setTimeout(() => train(++i), 10)
        } else {
            console.log('Done training, ' + i + ' iterations')
        }
    });
}