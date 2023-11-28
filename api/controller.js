const { predict } = require('./modelLoader');
const users = new Map();

const makePrediction = async (req, res) => {
  try {
    const prediction = await predict(req.body);
    res.json({ prediction });
  } catch (error) {
    console.error(error);
    res.status(500).send('Error in prediction');
  }
};

const createUser = (req, res) => {
  const { username, details } = req.body;
  if (users.has(username)) {
    res.status(400).send('User already exists');
  } else {
    users.set(username, details);
    res.status(200).send('User created successfully');
    console.log(users);
  }
};

module.exports = {
  makePrediction,
  createUser,
};
