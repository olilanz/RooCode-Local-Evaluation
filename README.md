# Roo Code - Autocoder

Roo Code demonstrates a truly fantastic approach to boost developer productivity by using cloud hosted LLMs to assist coding tasks. Cloud hosted LLMs like Claude Sonnet 3.7 are showing encouraging results, being context aware, and inserting useful code. Though at a relatively high price, making it economically questionable for exploratory/hobbyist tasks.

One of the benefits of tools like Roo Code is that they can also run against locally hosted LLMs, exposed through Ollama or other LLM runtimes, and thus make it runnable on consumer hardware. But how well does it work? Can you get more productive? Or is it just a hype? That's what I am exploring in this articele.

## Landscape

Nearly not a week goes by without new, truly new groundbreaking innovation being announce in the public space. The awesome new AI models the were chnagine the world upside-down last week, are replaces by even more capable models - even for free next week. GPT 4o, Deepseek R1, Grok, and so forth... In this fast changing environment, it is hard to stay on top with the latest development. 

Though, one thing remains true. Higher intelligence is not useful, if it cannot be applied. And unless we reliably solve the applicability problem of AI within a practical field, AI does not grow beyond a fascinating dream. 

The world is starting to understand that there will likely not be a single LLM that can solve it all for humatity. More likely, there will be numerous small models, carefully trained for particular purposes, and well orchestrated to fulfil difficult or tedious, but yet specific tasks in the real world. 

The industry starts to develop protocols like MCP to let the world develop specific AI agents, that can be combined into a larger autonomous workflow. That approach seems at this point the most promising, which is dubbed by technology leaders of Nvidia or Microsoft as "Agentic AI". This approach makes the training process of the smaller, more specialized models manegeable and economically feaible at smaller budgets, plus it confines advances and regressions in specific areas, allowing us to make controled advancements at mangeale quality.


## Auto-coders and Coding Assistants

The discipline of a software engineer can be seen as one of such specific areas where such AI agents can become useful. In that discipline we see a handful of very promising products emerging, such as Copilot, Windrurf, Continue, or Roo Code (formerly Roo Cline). 

The coding assistants like Copilot, that act more like a chat interface and provide very capable auto completion, emerge rapidly, and are often for free. They are becoming commonplace in a software engineer's toolbox, for rapid code analysis, development, review and debugging. 

The other category of tools, I refer here to as auto-coders, that can autonomously complete coding tasks, are a quite different beast. The engineer formulates a specific task for the autocoder, who then executes the task and provides the completed changes to the engineer for refinement and approval.

Autoconding requires much stronger context awareness, and higher precision, so a coding task can be carried out to completion. Now, there are tons of YouTube videos availale showing autocoders write a complete testris game from scratch within minutes, and a whole new paradigm called Vibe Coding starts to emerge. 

But more often than not, software engineers are working with large, existsing code bases, where years of though has gone into it to produce the desired outcome, concerning functionality, performance, security, robustness, or maintainability and the likes. In order to autonomously complete a task satisfacorily, the autocoder needs to infer from existing code the structure, intent, and quality aspects, before it can come up with a reasonable plan for making changes.

As LLMs are stateless, autocoders must solve concerns of context management, iterative refinement of understanding of the context, tool calling for knowledge acquisition and application of changes to the actual code base. This increased complexity sets it apart from coding assistants, and puts much more specific requirements on the LLMs capability. In other words, the LLMs needs specific training, fine-tuning, and extraordinary precision when exeuting the task. And particulary the issue of precision may be a challange with running LLMs on consumer hardware.

But then, can it be done? And if so, what tasks can it do?

## Why Roo Code?

Roo Code is  well established in the community, as it is super accessible as plug-in for VSCode, does a terrific job in integrating in the VSCode workflows, such as for applying code changes, executing commands directly in the VSCode terminal, integrating with source control, working with diffs, etc.

It can be configured to use local LLMs, and it is open source, which lets us explore in details why something is working and, and fix it if it doesnt. 

The thing that stands out though is that Roo Code lets you define custom roles in the context of the code base you are working with, e.g. for coding, as an architect, for asking questions fur building understanding. Choosing the role affets Roo Code's behavior, whih makes it feel like you're working with an colleague with a specialization and purpose. You can ask to get an understanding, get an architect's considerations, and get the coder to implement the changes to perfection.

While these default roles do a decent job, you can also create your own, e.g. as a helpful reviewer, a tech writer. You can furthermore specialize them to focus on specific quiality spects like robustness, performance or seurity - which as outlined earlier, I find important for real-world tasks.

Then, Roo Code also has an active community with frequent discussions on Discord, and frequent releases - which lowers the time needed between ideation to release of a new version of Roo Code.  

## Why doing this?

With 30 years of professional experience in the software industry, under constant technological change, one thing has been persitant. Refactoring production systems is hard. It challenges us engineers to maximum capacity. Now, here comes AI, and promises a truly helpul hand. It is a machine, never forgets, and should theoretically be the perfect companion for tedious tasks, or tasks where the human mind struggles to maintain complete overview. The machine, that never has a bad day, and always ready to help. The ultimate Swiss Army Knive. The willing code that help with everything, from speeding up your learning, remembering the semicolons, to running though millions of lines of log files, or testing out new strategies - while you are having lunch and are thinking of the next great thing to do. I think that a good autocoder can be really an improvement to a software engineer's quality of life.

At the same time, to me, it is not only about the end-state. With rapid development, it is no longer clear what the end-state should be. But to me it is much rather to also learn about the inner workings of tools like autocoders, as well as using them for exploratory things - even for seemingly silly ideas - without having to think of monetary footprint for every interaction - which frankly can really take the fun out of it.

And then lastly, but maybe even more importantly, resource constraints tend to accelerate the learning, as things do not tend to work out the way you'd think. You are presented with one challange at the time, which then forces you to understand the inner workings, and to consider different approaches. 

## The setup

For the work I entirley rely on pre-owned hardware components, which I purchased over the years, or for this particular purpose. It is not the most modern setup, but definitely on hogh end for consumer PCs. I use a previous gen AMD Ryzen 7, 128GB DDR4 RAM, 2TB NVMe, and 2 GPUs on PCIe 4: RTX 3090 TI and RTX 3060. This gives me a total of 36GB of VRAM to work with.

For the software I am using Unraid OS, as it us super lightwaight, and gives me painless support for containerization and virtualization with GPU pass-though. Initially I ran Windows and Ubuntu on Unraid using qemu for exploring around with Ollama, llama.cpp, Pinokio, LocalAI. But as I become more settled on the Ollama idea, I ditched the VMS, and run everything on containers on the Unraid. This keeps it light weight and make it all repeatable on other free platforms, such as Docker Compose or Kubernetes. 

Roo Code runs on the Mac, but with remote development on Unraid. So, essenially no software installed on the Mac - other than VSCode. All workloads and AI related processing is delegate to my Unraid server.

## How autocoders work

Debugging VSCode and the Roo Code plug-in to understand the inner workings can be a daunting task, as there is a lot of code running, that is not related to the LLM interactions. The way I found useful to build the understanding was to enable debug logging on Ollama, and to monitor the HTTP traffic between Roo Code and Ollama. 

Now, as mentioned indicated before, solving a complex task requires numerous, iterative LLM interactions. As the LLMs are stateless, and do neither have access to storage or tool calling, everything that the LLM needs to know to do the next task is included and sent to the LLM in every single call to the LLM. 

In Roo Code, such calls range from 200kb upwards, even if you just ask the model to explain afew lines of code. And iwth every iteration, the message grows in size, as more information is collected, and hence the context grows.

The message to the LLM, the prompt, contains everything: For example, the system prompt to explain how the model should behave and format responses - including what tools it can request to be called. The user prompt, which contains your request to the LLM, including the contents of any file that you reference. It also includes all the prompts from previous interactions with the LLM, so it can con decide what to do next, without starting all from the beginning again. 

The reply from the LLM, the assistant prompt, is streamed back to Roo Code, making one token appear at the time. Once Roo Code has received all tokens, it parses the message, and decides what to do next. The LLM may as for a tool call, such as read_file, e.g. if the LLM thinks that it is missing more information that may be conatined in a file that wasn't provided before. It may ask for changes to be made to a file, or it may ask for folder to be created. Roo will then ask you whether whether the request is reasonable, and then execute the request if the you approve - or ive you have configured auto-approval. The result of the tool call will then be embedded into the context for the next call to he LLM.

Now, as Roo Code supports nearly inifite combinations of LLMs and operting sytems with specific terminal capabilities, some tool calls may fail. Either because the LLM has made wrong assumptions, the LLM was not specific enough, or Roo Code's system prompt was too vague. To cater for such cases, Roo Code has a number of checks built in, which evaluate the LLM response, and send it back to the LLM with corrective instructions.

The less your model, operating system, system prompts and harware capabilities are aligned, the more you will observe these faults during task execution, and the more you will have Roo Code circle back to the LLM, and ask for re-evaluation. Eventually, the LLM and Roo Code have accumulated enough knowledge in the context, and will agree to make the code changes, complete the task that you have sett up for it, and present the output to you. 

Roo Code will now wrap it up, and leave it up to you to do the code review, and any submission of the altered code to the source code repo. you are now ready for the next task.

## Here is what is hard

Now, simple tasks, such as "remove all unreferenced private functions in class xyz" are easy, require few iterations, and are likely to exceed. Even intellectually difficult tasks like "improve my algorithm in function XYZ for performance at the expense of memory" can e carried our with ease - given that your LLM is trained to do this. 

When it comes to more involved tasks, such as "break down class XYZ for maintainability, and factor functionality ABC out into a separate class". Now, the LLM will not have all information needed, in one code, and it needs to create new files. In addition, it needs to gather more information about where the class is instantiated from, and what functions are publically used. The LLM and Roo Code will now work in iterations to make the step-by-step plan, gather the information, create the file structures, etc... all building up to a large context, being passed back and forth. 

Now, two interesting things happen: 

- Minor decisions in early iterations become become hard facts, as the evolving context is building up around eveything that has been learned. The effect of incorrect assumptions or imprecise formulations, is then amplified in subsequent iterations, as they now stand indispitedly in the context for the duration of the entire task. If the LLM is not trained or fine-tuned for precise ineractions for the task at hand, likelihood for the model wandering off into a wrong decision path and producing unusable code increases. Roo Code will not detect this, but you will be left to wonder what happend, undo all the chnages, and try with a simpler task instead.

- With increasing number of iterations, the physical size of the context is growing large - and eventually outgrows the model's context limitations. When that happens then, is that the model either truncates its response, or starts to become very loose when considering the provided context. That depends on the case. In best case, Roo Code will get upset, and decide to quit the task. In worse case, Roo Code will go on forever, and keep trying to get the model back on track, but making the matter with the conext size worse.

So, in essence, the simpler your task, the more chance you have for success. But the less you will gain from teh Roo Code setup. Subsequently, if you want to benefit most, you will want to find the best possible combination of model, model configuration, Roo Code system prompt, and hardware capabilities to suit your task. If all these parameters are fixed, you can start reducing task complexity by breaking down the tasks for Roo Code into smaller, more manageable junks.

This process can cause a lot of furstration, as there is no easy fix. Solutions are individual, and not much guidance is available on the internet. 

And this is prbably exactly why hosted solutions like Clause Sonnet 3.7 are attractive - despite their high costs. Their configuration is consitent, their model is fine-tuned with great precision, Roo Code has great system prompts and and is tested with it, and the backing hardware supports extremely large context. Essentlially, you get the best possible setup - obviously at the unfavourable price point. 

## Scaling up from your experience

Though, here is the cool part. Once you master the optimization on your local setup, you have the knowledge you need, so that can scale up using rented GPUs, through provides like runpod.io. There you can host your ollama back-end with your configuration on powerful GPUs for as little as half a dollar per hour, and have as many tokens pass back and forth at a flat rate. And when you're done, just shut it down, so you don't pay over night. 

Roo Code and the knowledge about local setup will definitely play well in that environment. 

# Setting the ambitions

## Speed vs Size

Most models can work purely on CPU, and off-load to GPU for faster processing. This makes it virtully possible to runn the largest models with the cost complicated tasks on cheap hardware - with enough system RAM and disk space. Though, for getting reasonable speeds, the model and the context should ideally fit entirely onto your GPU, as this reduces the amount of swapping of GPU memory (VRAM). Running inference tasks on the GPU instead of a combination of CPU and GPU litterally makes a speed difference of factor tens or hundreds.

For practical application with iterative refactoring of code, it is highly advisable to run 100% on GPU. So, when loading the configured model i Ollama, heck with 

´´´bash
ollama ps
´´´

To what extent your configuration makes use of the GPU.

## How many billions parameters?

Models come in sizes, which are determined during training of the model. Often you find modles in different sizes designated in their names or tags. 230b, 70b, 24b, 14b, 8b, 3b, 1.5b, etc... The more parameters in the model, the more nuanced it can be. 

My hardware, and 36GB VRAM, maxes out with models between 8b and 14b - dependent on what context size I use. I can run 24b models, but the the available space for context becomes impractically small. 

So, with number of parameters, go as large as your setup permits.

## Training

Now, when you choose the model, it's probably best if the billions of parameters were used during training for codign tasks. So the majority of the training should be done on on computerized languages (programming languages), and less on natural languages. There needs to be some training on natural language, as the LLM obviously needs to understand your requests, so it can perform the programming for you.

Often, the models for these tasks are designated in their names or tags with 'instruct' or 'coder' - fine tuned for following automated instructions, or specifically for coding tasks. But that is not a reliable measure though. So you need to read up on what models work well. 

Popular models are mistral, qwen2.5, phi4, codellama, deepseek-coder, falcon, and more. Some models are distilled from larger, general purpose models. Through destillation, a large model's knowledge can be used traing a smaller, more specific model, so it has similar or even better capabilities using less hardware. 

Finding a model that suits your task is your own challange. For my part, I seem to have best results using the qwen2.5 derived models.

## Quanrization

Next to the context, most of the VRAM utilization of a loaded model is used for the trained parameters, also called model weights, stored in multi-dinebsional arrays, also called tensors. Every parameter is represented using a datatype - during training, most commonly 32bit, as fp32. 64bit representations offer better precision, but are extremely uncommon outside of scientific realms.

Now, here is problem. If you choose a 24b model, with the training your that fits your need, and you run it at 32bit precision, your model alone will consume `24b * 32bit = 96GB` of VRAM. That does not even include the memory required for context. That amount of VRAM does hardly count as consumer-level hardware - you can rent them though at runpod.io at hourly rate, making it still accessible. 

Here is where quantization comes in. It is essentially the conversion of the parameters into smaller, lower precision data types. Very common are the 16bit representations in fp16 or bf16, at half precision - and therefore also half the required space. 

With my 36GB of VRAM, I need to go even lower, for example to 8bits, designated as Q8 or further down to Q6, Q5 or Q4. Now reduction of data precision of course affects the model's precision, and at too low precision, the model will start to halucinate easily, and the resulting refactoring task turns into comedy.

Now, this is where a balance needs to be struck between the models number of parameters, and the precision given for each parameter. 

In my case, with 36GB of VRAM, I see reasonable results with a Q8 quntization of a 14b model on Qwen training.

# Context size

Context size is the last, but probably most important thing to consider. While it is easy to imagine what it is, it is probably the least trivial to get right. So first a bit of explanation.

Context size refers to the number of tokens a model can process at a given point in time. You can call it as active memory for processing. A token is a sequence of bytes that was identified during model training - typically a word a stem, or some punctuation. The larger the contextis, the more complex the task can be that you expresse.

Now, the context is shared between the request you are sending to the LLM, as well as its response. In the use case of Roo Code, the messages being sent to the LLM can become very large for reasons discussed earlier. So on low context configurations, there might not be space enough for the result message, derailling any refactoring attempt. 

Now here is the challenge. While Ollama lets you freely choose the context size to suit your needs - or to max out your hardware - your LLM might not perform reliable on larger context sizes. If the model itself is trained with a particular context size, results can be arbitrary when exceeding that trained size. 

I have tried with varying results to go by 1.5 to 2.0 times larger than the trained context size. Things then become noatbly more inconsitent, and LLM and Roo Code seem to wander in circles. 

But how do you identify the trained context size, so you don't overshoot? Unfortunately, this in only knon during trainging and connot be seen on the model, except when it is attached to the model as metadata. Yhe cna see this in the Ollama log when loading the model.

Here is the example of the Unsloth/phi4 model:
```log
llm_load_print_meta: n_ctx_train      = 16384
```

The model was trained with a 16k context. So expanding that to 128k context will likely not make you happy.

Moreover, some models state support for larger context - although they have been trained for smaller context. What they do is applying a technique called role scaling. The essentially extrapolate at the expense of precision. Dependent on the arlgorthm, they scale linearly, or more pecide in the bower end and less prcise in the higher end of the context.

You can also see such properties i the ollama log (with my annotations):

```log
llama.context_length	16,384	# Base trained context length
llama.rope.dimension_count	128	# RoPE position encoding size
llama.rope.scaling.type	linear	# Uses linear RoPE scaling for extended context
llama.rope.scaling.factor	4.0	# Suggests it was extrapolated to 4× its trained length (16K × 4 = 64K)
```

So, what is the learning here? Consult the model's documentation, and verify the behavior in the ollama log for setting the initial context size. Bumping the context size up to max is likely not giving you the desired result. 

In my setup, on 36GB VRAM, I seem to find a good lee-way with the Qwen model, traind for the 1M context size, but sizing it down to 42k for fitting into my hardware. 

# Where to get the models from?

Well, naturally, as we are working with Ollama, the natural starting point is on http://ollama.org. However, far more combinations of training methods, quantizations, fine-tunings, distillations, can be found on Huggingface at http://hugginface.co. If you are new to this, start out with Ollama. It's well contained. Once you know what you need, move on to Huggingface. This is the Pandora's box - once you opened it, there is no turning back. 

For keeping track of my models, I create modelfiles in a folder, one for each model I am testing with. Then I have a small script that just rebuilds all models whenevery I make a change. I provide a few exampes here:

Example with Qwen from Huggingface.co:

```modelfile
# Very promising! Few minor mis-steps, but able to recover and continue
# 56k was slow but stable. 48k is fast, but crashed ollama worker process.
# Still testing.

# Prompt:
# I want you to improve maintainability of the Python code in @/inference/gradio_server.py . Organize the imports, and factor the args parsing out into a new file ultils/args.py. Add short documentation on every function to indicate the purpose of the function.

#FROM MHKetbi/Qwen2.5-Coder-32B-Instruct-Roo:q8_0
# 32GB VRAM at 56k context size.
# 51GB VRAM at 32k context size.

FROM hf.co/lmstudio-community/Qwen2.5-14B-Instruct-1M-GGUF:Q8_0
# Supports up to 1M context size.
# 40GB VRAM at 56k context size.
# PARAMETER num_ctx 57344
# 36GB VRAM at 48k context size. But crashes !?
# PARAMETER num_ctx 49125
# 34GB VRAM at 42k context size. But crashes !?
PARAMETER num_ctx 43008

#PARAMETER num_ctx 32768
#PARAMETER num_ctx 65536
```

Example with Deepseek R1 from Ollama.org:

```modelfile
# Very promising. Though keeps asking without "user_question". 
# The code is then written into the console instead if the file. 
# As a result, the files were almost empty.

# The model has an engineered context size max context size of 128k. However, it is not sure how well it will perform on the large end.

# With context size 32k, the model uses 36GB of VRAM
#FROM tom_himanen/deepseek-r1-roo-cline-tools:32b
#PARAMETER num_ctx 32768

# 33GB VRAM at 57 context size.
FROM tom_himanen/deepseek-r1-roo-cline-tools:14b
PARAMETER num_ctx 57344


#FROM tom_himanen/deepseek-r1-roo-cline-tools:8b

#PARAMETER num_ctx 65536
#PARAMETER num_ctx 100000
#PARAMETER num_ctx 131072
```

You can the get ollama to download all dependencies and create your own model with based on the base model, and your configured context size:

```bash
ollama create mymodel -f mymodelfile
```

The reason why I don't supply the context size during ollama start-up is that model files support many other parameters too, allow comments, and integrate easier with a source control workflow.  

If you do this at scale, you may find joy in my script for rebuilding all model files in the current folder. The model files must be named accorindg to the following standard: [mymodelnmae]-[mymodeltag].modefile

```bash
#! /usr/bin/bash

# Get the directory of the script
script_dir="$(dirname "$0")"

rebuild_model() {
    local filename="$1"
    local base_name="$(basename "${filename%.*}")"
    local model="${base_name%%-*}"
    local tag="${base_name#*-}"

    echo "Rebuilding model $model with tag $tag from file $filename..."

    ollama rm "$model:$tag"
    ollama create "$model:$tag" -f "$filename"
}

# Call rebuild_model for every *.modelfile in the script directory
for modelfile in "$script_dir"/*.modelfile; do
    if [ -e "$modelfile" ]; then
        rebuild_model "$modelfile"
    fi
```

# Testing the task

I have installed Roo Code, pionted it at my Ollama instance, with the following model configuration:

```modelfile
FROM hf.co/lmstudio-community/Qwen2.5-14B-Instruct-1M-GGUF:Q8_0
PARAMETER num_ctx 43008
```

This maxes my 36GB of VRAM almost out, and gives me the joy for a few simple tasks. 

I use the deepmeepbeep's YeEGP repo for testing, as I still have it lingering around from an earlier experiment: https://github.com/deepbeepmeep

And I run Roo Code over it using the folllowing prompt for the coding role: 

```quote
I want you to improve maintainability of the Python code in @/inference/gradio_server.py . Organize the imports, and factor the args parsing out into a new file ultils/args.py. Add short documentation on every function to indicate the purpose of the function."
```

I could definitely break that down into smaller tasks, and make it more specific. But I want to see if this task can be handled. It seems to me like a fair, real world thing to do. 

The task takes a iterations, b of them intentional, c are corrective due to model imprecision. It takes x minutes to complete. And the resulting code is useful and works as intended. 


# Conclusion

Getting to this point has not been straight forward. Luckily I am equipped with a healthy amount of curiosity, and a high threshold for pain. I could have chosen an easier setup

Though, I would never send the code as a pull request, as the model does not pay much attention to existsing structures and formatting, and you see code blocks flipping around.

All in all, I think I have a setup that works. Through fine-tuning of the sytem prompt, and more carefully wording my instructurions, I could probably reach a satisfying state. I am impressed.

Though, considering the amount of effort it has taken me to make this particular setup, and the speed at which new model evolve, and the setup likely will need revision, I am not sure whether I can wholeheartedly recommend that to everyone. 

In my case, where I like to tinker with tools for recreational purposes, maintaining that setup seems fine. But If I was in it purely for the productivity boot, I am not sure whether I could justify the hardware and time investments at this point. 

Taking the hit, and using Roo Code with a hosted provider may get you up and running faster and more predictably. 


