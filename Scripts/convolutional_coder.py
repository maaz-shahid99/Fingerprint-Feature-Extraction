import hashlib
from typing import List, Tuple, Union, Optional
import numpy as np

class RobustBiometricTemplateGenerator:
    """
    Improved biometric template generator with proper error handling,
    deterministic behavior, and security considerations.
    """
    
    def __init__(self, use_termination: bool = True, output_format: str = 'hash'):
        """
        Args:
            use_termination: Whether to append tail bits for proper termination
            output_format: 'decimal', 'binary', or 'hash' for output format
        """
        self.use_termination = use_termination
        self.output_format = output_format
        self.constraint_length = 3  # K = 3 (2 memory bits + current input)
        
    def quantize_angle(self, theta: float) -> Tuple[int, int]:
        """
        Robust angle quantization with explicit boundary handling.
        
        Quantization scheme (consistent left-exclusive, right-inclusive):
        [0,90] → (0,0)
        (90,180] → (0,1)  
        (180,270] → (1,0)
        (270,360) → (1,1)
        
        Args:
            theta: Angle in degrees
            
        Returns:
            Tuple of two integers (0 or 1)
            
        Raises:
            ValueError: If theta is not a finite number
        """
        if not isinstance(theta, (int, float)) or not np.isfinite(theta):
            raise ValueError(f"Angle must be a finite number, got {theta}")
        
        # Normalize to [0, 360) range
        normalized = float(theta) % 360.0
        
        # Handle the special case of -0.0
        if normalized == -0.0:
            normalized = 0.0
            
        if 0 <= normalized <= 90:
            return (0, 0)
        elif 90 < normalized <= 180:
            return (0, 1)
        elif 180 < normalized <= 270:
            return (1, 0)
        else:  # 270 < normalized < 360
            return (1, 1)
    
    def generate_codeword_from_angles(self, angle_list: List[float]) -> Tuple[str, Union[int, str]]:
        """
        Generate codeword from angle list with proper encoder handling.
        
        Args:
            angle_list: List of angles in degrees
            
        Returns:
            Tuple of (binary_string, output_value)
            where output_value is int, binary string, or hash depending on output_format
        """
        if not angle_list:
            return "", self._format_output("")
        
        # Create fresh encoder for deterministic results
        encoder = ConvolutionEncoder()
        
        try:
            # Step 1: Quantize all angles to bit tuples
            quantized_states = [self.quantize_angle(theta) for theta in angle_list]
            
            # Step 2: Flatten to bitstream
            bitstream = []
            for state in quantized_states:
                bitstream.extend(state)
            
            # Step 3: Add termination bits if enabled
            if self.use_termination:
                # Append K-1 zeros to flush the encoder
                bitstream.extend([0] * (self.constraint_length - 1))
            
            # Step 4: Encode using convolution encoder
            encoded_output = encoder.encode_sequence(bitstream)
            
            # Step 5: Convert to binary string
            codeword = ''.join([f'{x}{y}' for x, y in encoded_output])
            
            return codeword, self._format_output(codeword)
            
        except Exception as e:
            raise RuntimeError(f"Encoding failed: {e}")
    
    def _format_output(self, codeword: str) -> Union[int, str]:
        """Format output based on configuration."""
        if not codeword:
            return 0 if self.output_format == 'decimal' else ""
            
        if self.output_format == 'decimal':
            # Warn if very large
            if len(codeword) > 64:
                print(f"Warning: Large codeword ({len(codeword)} bits) may cause performance issues")
            return int(codeword, 2)
        elif self.output_format == 'hash':
            return hashlib.sha256(codeword.encode()).hexdigest()
        else:  # binary
            return codeword
    
    def process_minutiae_sets(self, ma_angles: List[float], 
                             mb_triplet_angles: List[float],
                             mc_triplet_angles: List[float], 
                             md_triplet_angles: List[float]) -> dict:
        """
        Process all four minutiae sets with independent, deterministic encoding.
        """
        results = {}
        
        # Process each set independently
        sets = {
            'Ma': ma_angles,
            'Mb': mb_triplet_angles, 
            'Mc': mc_triplet_angles,
            'Md': md_triplet_angles
        }
        
        for set_name, angles in sets.items():
            try:
                codeword, output_value = self.generate_codeword_from_angles(angles)
                results[f'g{set_name.lower()}'] = output_value
                results.setdefault('codewords', {})[set_name] = codeword
            except Exception as e:
                print(f"Error processing {set_name}: {e}")
                results[f'g{set_name.lower()}'] = None
                
        return results

class ConvolutionEncoder:
    """
    Rate-1/2 convolutional encoder with generator polynomials.
    
    Generator polynomials (in binary):
    G1 = 111 (octal 7) → X = C ⊕ B ⊕ A  
    G2 = 101 (octal 5) → Y = C ⊕ A
    
    Constraint length K = 3, memory order: [B, A] (newest to oldest)
    """
    
    def __init__(self):
        self.memory = [0, 0]  # [B, A] - shift register memory
        
    def encode_bit(self, c_bit: int) -> Tuple[int, int]:
        """
        Encode single bit with generator polynomials.
        
        Args:
            c_bit: Input bit (0 or 1)
            
        Returns:
            Tuple (X, Y) output bits
        """
        if c_bit not in (0, 1):
            raise ValueError(f"Input bit must be 0 or 1, got {c_bit}")
            
        b_bit, a_bit = self.memory
        
        # Apply generator polynomials
        x_out = c_bit ^ b_bit ^ a_bit  # G1 = 111
        y_out = c_bit ^ a_bit          # G2 = 101
        
        # Update shift register: [C, B] (shift right, insert new)
        self.memory = [c_bit, b_bit]
        
        return (x_out, y_out)
    
    def encode_sequence(self, bit_sequence: List[int]) -> List[Tuple[int, int]]:
        """Encode sequence of bits."""
        encoded_output = []
        for bit in bit_sequence:
            encoded_output.append(self.encode_bit(bit))
        return encoded_output
    
    def reset(self):
        """Reset encoder to initial state."""
        self.memory = [0, 0]
    
    def get_state(self) -> Tuple[int, int]:
        """Get current encoder state."""
        return tuple(self.memory)

# Comprehensive test suite
def test_boundary_cases():
    """Test critical boundary cases and edge conditions."""
    generator = RobustBiometricTemplateGenerator(output_format='binary')
    
    # Test boundary angles
    boundary_angles = [0, 90, 180, 270, 360, -0.0]
    expected_states = [(0,0), (0,0), (0,1), (1,0), (0,0), (0,0)]
    
    print("=== Boundary Angle Tests ===")
    for angle, expected in zip(boundary_angles, expected_states):
        result = generator.quantize_angle(angle)
        status = "✓" if result == expected else "✗"
        print(f"{status} Angle {angle:6.1f}° → {result} (expected {expected})")
    
    # Test extreme angles
    extreme_angles = [-360, -90, 450, 720, 90.0, 180.0, 270.0, 359.999]
    print("\n=== Extreme Angle Tests ===")
    for angle in extreme_angles:
        try:
            result = generator.quantize_angle(angle)
            print(f"✓ Angle {angle:8.3f}° → {result}")
        except Exception as e:
            print(f"✗ Angle {angle:8.3f}° → Error: {e}")
    
    # Test invalid inputs
    print("\n=== Invalid Input Tests ===")
    invalid_inputs = [float('inf'), float('nan'), None, 'not_a_number']
    for invalid in invalid_inputs:
        try:
            generator.quantize_angle(invalid)
            print(f"✗ {invalid} should have raised error")
        except (ValueError, TypeError):
            print(f"✓ {invalid} correctly rejected")
    
    # Test determinism across multiple calls
    print("\n=== Determinism Tests ===")
    test_angles = [45, 135, 225, 315]
    result1 = generator.process_minutiae_sets(test_angles, [], [], [])
    result2 = generator.process_minutiae_sets(test_angles, [], [], [])
    
    if result1['gma'] == result2['gma']:
        print("✓ Multiple calls produce identical results")
    else:
        print("✗ Non-deterministic behavior detected")
    
    # Test empty input
    print("\n=== Edge Case Tests ===")
    empty_result = generator.generate_codeword_from_angles([])
    print(f"✓ Empty input: {empty_result}")

def test_encoder_properties():
    """Test convolutional encoder properties."""
    print("\n=== Encoder Property Tests ===")
    
    encoder = ConvolutionEncoder()
    
    # Test state tracking
    initial_state = encoder.get_state()
    print(f"Initial state: {initial_state}")
    
    # Test sequence encoding
    test_sequence = [1, 0, 1, 1]
    output = encoder.encode_sequence(test_sequence)
    print(f"Input:  {test_sequence}")
    print(f"Output: {output}")
    print(f"Final state: {encoder.get_state()}")
    
    # Test reset
    encoder.reset()
    print(f"After reset: {encoder.get_state()}")

def demonstration():
    """Demonstrate the improved system."""
    print("\n=== System Demonstration ===")
    
    # Test with different output formats
    formats = ['binary', 'decimal', 'hash']
    
    for fmt in formats:
        print(f"\n--- Output Format: {fmt} ---")
        generator = RobustBiometricTemplateGenerator(output_format=fmt)
        
        # Sample biometric data
        ma_angles = [45, 135, 225, 315]
        mb_angles = [30, 120, 210, 300] 
        mc_angles = [60, 150, 240, 330]
        md_angles = [15, 105, 195, 285]
        
        results = generator.process_minutiae_sets(ma_angles, mb_angles, mc_angles, md_angles)
        
        for key, value in results.items():
            if key != 'codewords':
                print(f"{key}: {value}")

# Run all tests
if __name__ == "__main__":
    test_boundary_cases()
    test_encoder_properties()
    demonstration()
