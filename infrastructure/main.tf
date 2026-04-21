terraform {
  required_providers {
    azurerm = {
      source  = "hashicorp/azurerm"
      version = "~> 3.0"
    }
  }
}

provider "azurerm" {
  features {}
}

# -------------------------------------------------------------
# 1. Groupe de ressources & Réseau Sécurisé
# -------------------------------------------------------------
resource "azurerm_resource_group" "rg" {
  name     = "rg-total-seismic-agent"
  location = "Sweden Central" 
}

resource "azurerm_virtual_network" "vnet" {
  name                = "vnet-seismic"
  address_space       = ["10.0.0.0/16"]
  location            = azurerm_resource_group.rg.location
  resource_group_name = azurerm_resource_group.rg.name
}

resource "azurerm_subnet" "subnet_compute" {
  name                 = "snet-compute"
  resource_group_name  = azurerm_resource_group.rg.name
  virtual_network_name = azurerm_virtual_network.vnet.name
  address_prefixes     = ["10.0.1.0/24"]
}

# -------------------------------------------------------------
# 2. Instance d'Orchestration (Serveur LangGraph)
# -------------------------------------------------------------
resource "azurerm_network_interface" "nic_orch" {
  name                = "nic-orchestrator"
  location            = azurerm_resource_group.rg.location
  resource_group_name = azurerm_resource_group.rg.name

  ip_configuration {
    name                          = "internal"
    subnet_id                     = azurerm_subnet.subnet_compute.id
    private_ip_address_allocation = "Dynamic"
  }
}

resource "azurerm_linux_virtual_machine" "vm_orch" {
  name                = "vm-orchestrator-d8"
  resource_group_name = azurerm_resource_group.rg.name
  location            = azurerm_resource_group.rg.location
  size                = "Standard_D8s_v5"
  admin_username      = "azureuser"
  
  network_interface_ids = [azurerm_network_interface.nic_orch.id]

  admin_ssh_key {
    username   = "azureuser"
    public_key = var.ssh_public_key
  }

  os_disk {
    caching              = "ReadWrite"
    storage_account_type = "Premium_LRS"
  }

  source_image_reference {
    publisher = "Canonical"
    offer     = "0001-com-ubuntu-server-jammy"
    sku       = "22_04-lts"
    version   = "latest"
  }
}

# -------------------------------------------------------------
# 3. Instance GPU (Nvidia T4 pour validation JAX/FNO)
# -------------------------------------------------------------
resource "azurerm_network_interface" "nic_gpu" {
  name                = "nic-gpu"
  location            = azurerm_resource_group.rg.location
  resource_group_name = azurerm_resource_group.rg.name

  ip_configuration {
    name                          = "internal"
    subnet_id                     = azurerm_subnet.subnet_compute.id
    private_ip_address_allocation = "Dynamic"
  }
}

resource "azurerm_linux_virtual_machine" "vm_gpu" {
  name                = "vm-gpu-t4" # Renommée pour plus de clarté
  resource_group_name = azurerm_resource_group.rg.name
  location            = azurerm_resource_group.rg.location
  size                = "Standard_NC4as_T4_v3" # Économique (environ 0.50€/h)
  admin_username      = "azureuser"
  
  network_interface_ids = [azurerm_network_interface.nic_gpu.id]

  admin_ssh_key {
    username   = "azureuser"
    public_key = var.ssh_public_key
  }

  os_disk {
    caching              = "ReadWrite"
    storage_account_type = "Premium_LRS"
    disk_size_gb         = 512 # Réduit pour la PoC
  }

  source_image_reference {
    publisher = "microsoft-dsvm"
    offer     = "ubuntu-2204"
    sku       = "2204-gen2"
    version   = "latest"
  }
}

# --- SÉCURITÉ : Arrêt automatique quotidien à 19h00 ---
resource "azurerm_dev_test_global_vm_shutdown_schedule" "shutdown_gpu" {
  virtual_machine_id = azurerm_linux_virtual_machine.vm_gpu.id
  location           = azurerm_resource_group.rg.location
  enabled            = true

  daily_recurrence_time = "1900"
  timezone              = "Romance Standard Time"

  notification_settings {
    enabled = false
  }
}

# --- SÉCURITÉ : Arrêt automatique de l'orchestrateur à 19h00 ---
resource "azurerm_dev_test_global_vm_shutdown_schedule" "shutdown_orch" {
  virtual_machine_id = azurerm_linux_virtual_machine.vm_orch.id
  location           = azurerm_resource_group.rg.location
  enabled            = true

  daily_recurrence_time = "1900"
  timezone              = "Romance Standard Time"

  notification_settings {
    enabled = false
  }
}

# -------------------------------------------------------------
# 4. Azure AI Service pour Mistral Large v3
# -------------------------------------------------------------
resource "azurerm_ai_services" "ai_studio" {
  name                = "ai-hub-seismic"
  location            = azurerm_resource_group.rg.location
  resource_group_name = azurerm_resource_group.rg.name
  sku_name            = "S0"
}

resource "azurerm_cognitive_deployment" "mistral_large" {
  name                 = "mistral-large-v3"
  cognitive_account_id = azurerm_ai_services.ai_studio.id

  model {
    format  = "Mistral" # Correction : Mistral n'est pas au format OpenAI
    name    = "Mistral-large-2411" # Nom de modèle plus précis pour le catalogue Azure
    version = "1"
  }

  scale {
    type     = "Standard"
    capacity = 1
  }
}