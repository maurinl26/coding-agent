terraform {
  required_providers {
    azurerm = {
      source  = "hashicorp/azurerm"
      version = "~> 3.0"
    }
    azapi = {
      source  = "azure/azapi"
      version = "~> 1.13" 
    }
  }
}

provider "azurerm" {
  features {}
}

provider "azapi" {}

# -------------------------------------------------------------
# 1. Groupe de ressources & Réseau Sécurisé (Région esquivée)
# -------------------------------------------------------------
resource "azurerm_resource_group" "rg" {
  name     = "rg-total-seismic-agent"
  location = "East US" # Ou West Europe, là où les quotas sont ouverts
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
# 2. Instance de Calcul (GPU T4 & Outils Fortran)
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
  name                = "vm-gpu-t4-fortran"
  resource_group_name = azurerm_resource_group.rg.name
  location            = azurerm_resource_group.rg.location
  # Standard_B2s while GPU quota (Standard_NC4as_T4_v3) is pending approval
  size                = "Standard_B2s"
  admin_username      = "azureuser"
  
  network_interface_ids = [azurerm_network_interface.nic_gpu.id]

  admin_ssh_key {
    username   = "azureuser"
    public_key = var.ssh_public_key
  }

  os_disk {
    caching              = "ReadWrite"
    storage_account_type = "Premium_LRS"
    disk_size_gb         = 128 
  }

  source_image_reference {
    publisher = "Canonical"
    offer     = "0001-com-ubuntu-server-jammy"
    sku       = "22_04-lts"
    version   = "latest"
  }

  custom_data = base64encode(<<-EOF
    #!/bin/bash
    apt-get update && apt-get upgrade -y
    apt-get install -y build-essential wget curl python3-venv python3-pip
  EOF
  )
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

# -------------------------------------------------------------
# 3. Azure AI Service pour Mistral Large v3 (via AzAPI)
# -------------------------------------------------------------
resource "azurerm_ai_services" "ai_studio" {
  name                = "ai-hub-seismic"
  location            = azurerm_resource_group.rg.location
  resource_group_name = azurerm_resource_group.rg.name
  sku_name            = "S0"
}

resource "azapi_resource" "mistral_large" {
  type      = "Microsoft.CognitiveServices/accounts/deployments@2023-05-01"
  name      = "mistral-large-v3"
  parent_id = azurerm_ai_services.ai_studio.id
  body = jsonencode({
    sku = {
      name     = "Standard"
      capacity = 1
    }
    properties = {
      model = {
        format  = "Mistral"
        name    = "Mistral-large-2411"
        version = "1"
      }
    }
  })
}