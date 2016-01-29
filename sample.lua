local torch = require 'torch'
local nn = require 'nn'
require 'nngraph'
local MDN = require 'mdn'
local model = require 'model'
local model_utils = require 'model_utils'
local data = require 'data'

cmd = torch.CmdLine()
cmd:option('-model', 'model.t7','path to the input model file')
cmd:option('-seed', 123, 'random number generator seed')
cmd:option('-length', 1000, 'number of frames to generate')
cmd:option('-output', 'test.wav', 'path to the output audio file')
cmd:option('-bias', 1, 'sample variance scalar')
cmd:option('-gpu_id', -1, 'ID of the GPU to use (-1 for CPU)')
opt = cmd:parse(arg)

torch.manualSeed(opt.seed)

if opt.gpu_id >= 0 then
    local ok, cunn = pcall(require, 'cunn')
    local ok2, cutorch = pcall(require, 'cutorch')
    if not ok then print('package cunn not found!') end
    if not ok2 then print('package cutorch not found!') end
    if ok and ok2 then
        print('using CUDA on GPU ' .. opt.gpu_id .. '...')
        cutorch.setDevice(opt.gpu_id + 1) -- note +1 to make it 0 indexed! sigh lua
        cutorch.manualSeed(opt.seed)
    else
        print('If cutorch and cunn are installed, your CUDA toolkit may be improperly configured.')
        print('Check your CUDA toolkit installation, rebuild cutorch and cunn, and try again.')
        print('Falling back on CPU mode')
        opt.gpu_id = -1 -- overwrite user setting
    end
end

m = torch.load(opt.model)
print "post load model"

local protos = m.model.protos
local lstm_state = model_utils.clone_list(m.model.initstate)
for i = 1, #lstm_state do
    lstm_state[i] = torch.Tensor(1, lstm_state[i]:size(2)):copy(lstm_state[i][{1, {}}])
end

-- seeding
m.input.current_batch = 0
local x, _ = data.next_batch(m.input)

if opt.gpu_id >= 0 then
   x = x:cuda()
end

print "seeding"

for t = 0, m.params.seq_length * 10 do
    local i = (t % m.params.seq_length) + 1

    local next_state = protos.lstm:forward{
       torch.Tensor(1, x:size(3)):copy(x[{1, i, {}}]),
       unpack(lstm_state)
    }

    protos.linear_out:forward(next_state[#next_state])
    lstm_state = next_state
end

print "here too"

-- sequence generation
local sequence = {[0]=torch.zeros(1, m.params.input_size)}

for t = 1, opt.length do
    local next_state = protos.lstm:forward{
        torch.Tensor(1, x:size(3)):copy(sequence[t - 1]),
        unpack(lstm_state)
    }
    local probs = protos.linear_out:forward(next_state[#next_state])
    sequence[t] = protos.criterion:sample(probs, opt.bias):clone()
    lstm_state = next_state
end

data.synthesise(sequence, m.params, opt.output)
