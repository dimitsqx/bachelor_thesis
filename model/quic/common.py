import asyncio
import pickle


async def send_model_weights(weights, reader, writer):
    '''Send the Model weights
    '''
    weights_binary = pickle.dumps(weights)
    # Send model Length
    writer.write(str(len(weights_binary)).encode() + b'\n')
    # Send the weights
    # print(len(weights_binary))
    line = await reader.readline()
    # print(line)
    if line == b'Got pickle length\n':
        # print('in ifff')
        writer.write(weights_binary)
        # print('sent weights')
    line = await reader.readline()
    # print(line)
    if line == b'Got Model Weights\n':
        return True


async def get_model_weights(reader, writer):
    '''Routine to get model weights
    '''
    # Read message length
    line = await reader.readline()
    # print(line)
    writer.write(b'Got pickle length\n')
    # Read the weights
    # print('Wait for weights')
    weights_binary = await reader.readexactly(int(line))
    model_weights = pickle.loads(weights_binary)
    writer.write(b'Got Model Weights\n')
    return model_weights