function importJSON(){
    let jsonFile = document.getElementById('loadConfig').files[0];

    let reader = new FileReader();
    reader.readAsText(jsonFile, "UTF-8");
    reader.onload = function (evt) {
         const jsonConfig = JSON.parse(evt.target.result);
        setStuffFromObj(jsonConfig);
    }
}

function setStuffFromObj(jsonConfig){
    document.getElementById("base_model").value = jsonConfig.base_model;
    document.getElementById("img_folder").value = jsonConfig.img_folder;
    document.getElementById("output_folder").value = jsonConfig.output_folder;
    document.getElementById("save_json_folder").value = jsonConfig.save_json_folder;
    document.getElementById("save_json_name").value = jsonConfig.save_json_name;
    document.getElementById("training_comment").value = jsonConfig.training_comment;
    document.getElementById("scheduler").value = jsonConfig.scheduler;
    document.getElementById("cosine_restarts").value = jsonConfig.cosine_restarts;
    document.getElementById("scheduler_power").value = jsonConfig.scheduler_power;
    document.getElementById("learning_rate").value = jsonConfig.learning_rate;
    document.getElementById("unet_lr").value = jsonConfig.unet_lr;
    document.getElementById("text_encoder_lr").value = jsonConfig.text_encoder_lr;
    document.getElementById("unet_only").checked = jsonConfig.unet_only;
    document.getElementById("net_dim").value = jsonConfig.net_dim;
    document.getElementById("alpha").value = jsonConfig.alpha;
    document.getElementById("train_resolution").value = jsonConfig.train_resolution;
    document.getElementById("height_resolution").value = jsonConfig.height_resolution;
    document.getElementById("batch_size").value = jsonConfig.batch_size;
    document.getElementById("clip_skip").value = jsonConfig.clip_skip;
    document.getElementById("mixed_precision").value = jsonConfig.mixed_precision;
    document.getElementById("save_precision").value = jsonConfig.save_precision;
    document.getElementById("lyco").checked = jsonConfig.lyco;
    document.getElementById("conv_dim").value = jsonConfig.conv_dim;
    document.getElementById("conv_alpha").value = jsonConfig.conv_alpha;
    document.getElementById("use_conv_cp").checked = jsonConfig.use_conv_cp;
    document.getElementById("num_epochs").value = jsonConfig.num_epochs;
    document.getElementById("save_every_n_epochs").value = jsonConfig.save_every_n_epochs;
    document.getElementById("text_only").checked = jsonConfig.text_only;
}

function exportJSON(){
    let out = {
        "base_model": document.getElementById("base_model").value,
        "img_folder": document.getElementById("img_folder").value,
        "output_folder": document.getElementById("output_folder").value,
        "save_json_folder": document.getElementById("save_json_folder").value,
        "save_json_name": document.getElementById("save_json_name").value,
        "load_json_path": null,
        "multi_run_folder": null,
        "reg_img_folder": null,
        "sample_prompts": null,
        "change_output_name": document.getElementById("save_json_name").value,
        "json_load_skip_list": null,
        "training_comment": document.getElementById("training_comment").value,
        "save_json_only": false,
        "tag_occurrence_txt_file": true,
        "sort_tag_occurrence_alphabetically": false,
        "optimizer_type": "AdamW8bit",
        "optimizer_args": {
        "weight_decay": "0.1",
            "betas": "0.9,0.99"
        },
        "scheduler": document.getElementById("scheduler").value,
        "cosine_restarts": parseInt(document.getElementById("cosine_restarts").value),
        "scheduler_power": parseInt(document.getElementById("scheduler_power").value),
        "lr_scheduler_type": null,
        "lr_scheduler_args": null,
        "learning_rate": parseFloat(document.getElementById("learning_rate").value),
        "unet_lr": parseFloat(document.getElementById("unet_lr").value),
        "text_encoder_lr": parseFloat(document.getElementById("text_encoder_lr").value),
        "warmup_lr_ratio": null,
        "unet_only": document.getElementById("unet_only").checked,
        "net_dim": parseInt(document.getElementById("net_dim").value),
        "alpha": parseFloat(document.getElementById("alpha").value),
        "train_resolution": parseInt(document.getElementById("train_resolution").value),
        "height_resolution": parseInt(document.getElementById("height_resolution").value),
        "batch_size": parseInt(document.getElementById("batch_size").value),
        "clip_skip": parseInt(document.getElementById("clip_skip").value),
        "test_seed": 23,
        "mixed_precision": document.getElementById("mixed_precision").value,
        "save_precision": document.getElementById("save_precision").value,
        "lyco": document.getElementById("lyco").checked,
        "network_args": {
            "algo": "lora",
            "conv_dim": parseInt(document.getElementById("conv_dim").value),
            "conv_alpha": parseInt(document.getElementById("conv_alpha").value),
            "use_conv_cp": document.getElementById("use_conv_cp").checked
        },
        "num_epochs": parseInt(document.getElementById("num_epochs").value),
        "save_every_n_epochs": parseInt(document.getElementById("save_every_n_epochs").value),
        "save_n_epoch_ratio": null,
        "save_last_n_epochs": null,
        "max_steps": null,
        "sample_sampler": null,
        "sample_every_n_steps": null,
        "sample_every_n_epochs": null,
        "buckets": true,
        "min_bucket_resolution": 320,
        "max_bucket_resolution": 960,
        "bucket_reso_steps": null,
        "bucket_no_upscale": false,
        "shuffle_captions": false,
        "keep_tokens": null,
        "token_warmup_step": null,
        "token_warmup_min": null,
        "xformers": true,
        "cache_latents": false,
        "random_crop": true,
        "flip_aug": false,
        "v2": false,
        "v_parameterization": false,
        "gradient_checkpointing": false,
        "gradient_acc_steps": null,
        "noise_offset": null,
        "mem_eff_attn": false,
        "min_snr_gamma": null,
        "huggingface_repo_id": null,
        "huggingface_repo_type": null,
        "huggingface_path_in_repo": null,
        "huggingface_token": null,
        "huggingface_repo_visibility": null,
        "save_state_to_huggingface": false,
        "resume_from_huggingface": false,
        "async_upload": false,
        "lora_model_for_resume": null,
        "save_state": false,
        "resume": null,
        "text_only": document.getElementById("text_only").checked,
        "vae": null,
        "log_dir": null,
        "log_prefix": null,
        "tokenizer_cache_dir": null,
        "dataset_config": null,
        "lowram": false,
        "no_meta": false,
        "color_aug": false,
        "use_8bit_adam": false,
        "use_lion": false,
        "caption_dropout_rate": null,
        "caption_dropout_every_n_epochs": null,
        "caption_tag_dropout_rate": null,
        "prior_loss_weight": 1,
        "max_grad_norm": 1,
        "save_as": "safetensors",
        "caption_extension": ".txt",
        "max_clip_token_length": 150,
        "save_last_n_epochs_state": null,
        "num_workers": 1,
        "persistent_workers": true,
        "face_crop_aug_range": null,
        "network_module": "sd_scripts.networks.lora",
        "locon_dim": null,
        "locon_alpha": null,
        "locon": false,
        "list_of_json_to_run": null
    };

    let element = document.createElement('a');
    element.setAttribute('href', 'data:text/plain;charset=utf-8,' + encodeURIComponent(JSON.stringify(out, null, 4)));
    element.setAttribute('download', out.save_json_name + ".json");

    element.style.display = 'none';
    document.body.appendChild(element);

    element.click();

    document.body.removeChild(element);
}