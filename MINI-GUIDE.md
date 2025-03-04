# Quick-start guide to get Roo Code up and running with Ollama

This short guide explains how to get a basic configuration up and running with a consumer GPU with 8 to 12 GB of VRAM. 

The guide assumes that you have Ollama and Roo Code installed. Find instructions for doing so [here] and [here].

Make sure that you have access to Ollama from your command line.

## Loading the model into Ollama

Create a file called `mymodel.txt`, and copy the following contents:

```txt
FROM deepseek-coder:6.7B-instruct
PARAMETER num_ctx 16384
PARAMETER num_predict 8192
```

This file is used to provide Ollama with a few specifications for creating a new model. You can later change the parameters by extending the file as your needs evolve.

Now, open the terminal, and create your model using the following command:

```bash
ollama create mymodel -f mymodel.txt
```

This instructs Ollama to apply the specifications in your modelfile.txt, and and use it to create a new model called `mymodel`, wich you then can use for Roo Code. This may take a few minutes if you are running this for the first time, as Ollama will need to download the deepseek-coder model to your computer.

```bash
ollama run mymodel "Hello, are you there?"
```

As deepseek-coder does not possess much humor, it will likely remind you in a friendly tone that it prefers to work with code instead of doing small talk. In such case, everything is well.


## Configuring Roo Code to use the model

In the "Settings" tab, select Ollama as the API Provider. If Ollama is running on your own computer with a default configuration, you will not need to specify the the Base URL, as Roo Code defaults to http://localhost:11434. If you are not running Ollama somewhere else, specify the address.

Roo Code will automatically show a list of available models, if the connection to Ollama can be established.

You should now be able to see at lease two models: `mymodel` and `deepseek-coder:6.7B-instruct`. Place the check-mark at the `mymodel`

## Turning off MCP server support

Roo Code is in principle ready now for generating your first code. But before we do that, we need to turn off the MCP support, as this will not work in our minimal setup. To do so, click on the MCP Server icon, and make sure that the Use MCP Server check-box is un-checked.

## You are now ready to generate code

You can try something along those lines:

```userprompt
Create a minimal game canvas using Phaser.js (version 3.50 from a public CDN). The canvas should display a moving shape that bounces off the viewport bounds using real physics. The shape should change course when the space bar is hit.

Write the complete code to a single index.html file using the write_to_file tool, without using CDATA tags.
```

It may require a few attempts for the model to get this right. Don't hesitate to throw the generated code away if you don't like it, and try again. 

This prompt should generate a fully functional demo that you can load into a browser and enjoy.

## Explaining the model file

The parameters in the model file have been selected super conservatively, so that code generation completes within 1 or 2 minutes, even with less than 12 GB VRAM. If you have 8GB or less, the waiting time wil be a bit longer.

Here is the explanation for each line: 

`FROM deepseek-coder:6.7B-instruct` lets Ollama know which model we want to use as the base model when creating our own model. This particular model has 6.7 trained parameters and uses a 4-bit quantization, which is definitely on the low end for code generation tasks. If you have capable hardware, you may choose more powerful models, from the Ollama or Huggingface website. 

`PARAMETER num_ctx 16384` configures Ollama to allocate enough memory for 16k context. With increasing task complexity, you will need more memory for context. If your hardware allows, you can experiment with increasing this value.

`PARAMETER num_predict 8192` specifies the number of result tokens that the model may generate on your request. As Ollama may come with low default values, we specify the size explicitly. If you want to generate longer files, you can increase the number.

## Scaling up

If you have a lot of patience, or you have ample of VRAM, you can adjust the parameters as you see fit. With increased parameters, the complexity of tasks that Roo Code can successfully solve for you will increase. 

With the conservative settings in this example, you will quickly find the limits of the model, and code generation may fail - or end with broken code. If you have appetite for pushing the capabilities a notch up, you can try the following configuration: 

```txt
FROM hf.co/lmstudio-community/phi-4-GGUF:Q8_0
PARAMETER num_ctx 32768
PARAMETER num_predict 12000
```

There are many configurations that work. Finding the right one for your setup can be a tricky task though. Detailed information can be found here: [README.md](README.md)